import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import hashlib
import json
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, Dropout, GRU, Input, LSTM, LayerNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class LSTMPredictor:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns = config["MODEL_CONFIG"]["features"]
        self.sequence_length = config["MODEL_CONFIG"]["sequence_length"]

    def _interval_settings(self, interval: str) -> Dict[str, float]:
        interval = (interval or "1h").strip().lower()
        base = {
            "sequence_length": self.config["MODEL_CONFIG"]["sequence_length"],
            "batch_size": self.config["MODEL_CONFIG"]["batch_size"],
            "epochs": self.config["MODEL_CONFIG"]["epochs"],
            "learning_rate": self.config["MODEL_CONFIG"]["learning_rate"],
            "train_test_split": self.config["MODEL_CONFIG"]["train_test_split"],
            "fine_tune_learning_rate": self.config["MODEL_CONFIG"].get("fine_tune_learning_rate", 0.00015),
            "fine_tune_epochs": self.config["MODEL_CONFIG"].get("fine_tune_epochs", 6),
            "min_training_rows": self.sequence_length + 24,
            "recent_tune_window": 252,
            "accuracy_floor_pct": 95.0,
            "allowed_accuracy_drop_pct": 1.5,
            "consecutive_drop_limit": 3,
        }
        overrides = self.config["MODEL_CONFIG"].get("interval_overrides", {}).get(interval, {})
        return {**base, **overrides}

    def get_interval_settings(self, interval: str = "1h") -> Dict[str, float]:
        return self._interval_settings(interval)

    def get_sequence_length(self, interval: str = "1h") -> int:
        return int(self._interval_settings(interval)["sequence_length"])

    def _sanitize_interval(self, interval: str) -> str:
        return (interval or "1h").strip().lower().replace("/", "_")

    def model_dir(self, symbol: str, interval: str = "1h") -> str:
        safe_interval = self._sanitize_interval(interval)
        return f"models/saved/{symbol}/{safe_interval}"

    def model_path(self, symbol: str, interval: str = "1h") -> str:
        return f"{self.model_dir(symbol, interval)}/lstm_model.h5"

    def _legacy_model_dir(self, symbol: str) -> str:
        return f"models/saved/{symbol}"

    def has_model(self, symbol: str, interval: str = "1h") -> bool:
        if os.path.exists(self.model_path(symbol, interval)):
            return True
        return interval == "1h" and os.path.exists(f"{self._legacy_model_dir(symbol)}/lstm_model.h5")

    def prepare_features(self, df, interval: str = "1h"):
        """Prepare technical indicators and engineered features."""
        df = df.sort_values("timestamp").copy()

        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()

        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        sma = df["close"].rolling(window=20).mean()
        std = df["close"].rolling(window=20).std()
        df["bollinger_upper"] = sma + (std * 2)
        df["bollinger_lower"] = sma - (std * 2)

        df["return_1"] = df["close"].pct_change()
        df["return_5"] = df["close"].pct_change(periods=5)
        df["volatility_10"] = df["return_1"].rolling(window=10).std()
        df["momentum_10"] = df["close"] / df["close"].shift(10) - 1.0
        df["price_to_sma_20"] = df["close"] / df["sma_20"] - 1.0
        df["price_to_ema_12"] = df["close"] / df["ema_12"] - 1.0
        df["volume_ratio_10"] = df["volume"] / df["volume"].rolling(window=10).mean()
        df["high_low_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
        df["open_close_range"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)

        df = df.replace([np.inf, -np.inf], np.nan).bfill().ffill()
        return df[self.feature_columns]

    def create_sequences(self, data, target_col="close", sequence_length: Optional[int] = None):
        """Create sequences for recurrent training."""
        sequence_length = int(sequence_length or self.sequence_length)
        if isinstance(data, pd.DataFrame):
            values = data.to_numpy(dtype=np.float32, copy=True)
            target_idx = data.columns.get_loc(target_col)
        else:
            values = np.asarray(data, dtype=np.float32)
            if values.ndim != 2:
                raise ValueError("Sequence input must be a 2D array.")
            if isinstance(target_col, str):
                raise ValueError("target_col must be an integer index when data is not a DataFrame.")
            target_idx = int(target_col)

        X, y = [], []
        for i in range(sequence_length, len(values)):
            X.append(values[i - sequence_length:i])
            y.append(values[i, target_idx])

        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

    def _compile_model(self, model: Sequential, learning_rate: float):
        model.compile(
            optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss=tf.keras.losses.Huber(delta=1.0),
            metrics=["mae"],
        )
        return model

    def build_model(self, input_shape, interval: str = "1h"):
        """Build an interval-aware recurrent model."""
        settings = self._interval_settings(interval)
        width = 64 if interval == "1d" else 48
        tail_width = 48 if interval == "1d" else 32
        dropout_rate = 0.15 if interval == "1d" else 0.2

        model = Sequential(
            [
                Input(shape=input_shape),
                Conv1D(filters=32, kernel_size=3, padding="causal", activation="relu"),
                LayerNormalization(),
                Bidirectional(LSTM(width, return_sequences=True)),
                Dropout(dropout_rate),
                GRU(tail_width),
                Dropout(dropout_rate),
                Dense(48, activation="relu"),
                Dense(1),
            ]
        )
        return self._compile_model(model, learning_rate=float(settings["learning_rate"]))

    def _inverse_close_values(self, values, feature_count: int, close_idx: int):
        reshaped = np.asarray(values, dtype=np.float32).reshape(-1)
        dummy_array = np.zeros((len(reshaped), feature_count), dtype=np.float32)
        dummy_array[:, close_idx] = reshaped
        inverse_scaled = self.scaler.inverse_transform(dummy_array)
        return inverse_scaled[:, close_idx]

    def _calculate_regression_metrics(self, actual, predicted):
        actual = np.asarray(actual, dtype=np.float32).reshape(-1)
        predicted = np.asarray(predicted, dtype=np.float32).reshape(-1)
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        mae = float(mean_absolute_error(actual, predicted))
        denominator = np.where(np.abs(actual) < 1e-9, np.nan, np.abs(actual))
        mape = float(np.nanmean(np.abs((actual - predicted) / denominator)) * 100.0)
        accuracy_pct = float(max(0.0, 100.0 - mape)) if not np.isnan(mape) else None
        return {
            "rmse": rmse,
            "mae": mae,
            "mape": None if np.isnan(mape) else mape,
            "accuracy_pct": accuracy_pct,
        }

    def train(
        self,
        df,
        symbol,
        interval: str = "1h",
        continue_training: bool = False,
        recent_window: Optional[int] = None,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ):
        """Train or fine-tune the recurrent model."""
        settings = self._interval_settings(interval)
        sequence_length = int(settings["sequence_length"])
        features_df = self.prepare_features(df, interval=interval)
        minimum_rows = max(sequence_length + 24, int(settings["min_training_rows"]))
        if len(features_df) < minimum_rows:
            raise ValueError(
                f"Not enough rows to train {symbol} {interval}. Need at least {minimum_rows}, found {len(features_df)}."
            )

        if recent_window:
            trimmed_rows = max(sequence_length + 20, int(recent_window))
            features_df = features_df.tail(trimmed_rows).reset_index(drop=True)

        previous_metadata = self.load_metadata(symbol, interval) if continue_training and self.has_model(symbol, interval) else {}
        if continue_training and self.has_model(symbol, interval):
            self.load_model(symbol, interval)
            scaled_data = self.scaler.transform(features_df)
        else:
            self.scaler = MinMaxScaler()
            scaled_data = self.scaler.fit_transform(features_df)

        scaled_df = pd.DataFrame(scaled_data, columns=features_df.columns)
        X, y = self.create_sequences(scaled_df, sequence_length=sequence_length)
        if len(X) < 10:
            raise ValueError(
                f"Not enough training sequences for {symbol} {interval}. Need at least 10, found {len(X)}."
            )

        split_idx = int(len(X) * float(settings["train_test_split"]))
        split_idx = min(max(split_idx, 1), len(X) - 1)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if continue_training and self.model is not None:
            self._compile_model(self.model, learning_rate=float(learning_rate or settings["fine_tune_learning_rate"]))
        else:
            self.model = self.build_model((X_train.shape[1], X_train.shape[2]), interval=interval)
            if learning_rate is not None:
                self._compile_model(self.model, learning_rate=float(learning_rate))

        fit_epochs = int(epochs or (settings["fine_tune_epochs"] if continue_training else settings["epochs"]))
        fit_batch_size = int(settings["batch_size"])
        fit_kwargs = {
            "batch_size": fit_batch_size,
            "epochs": fit_epochs,
            "verbose": 0,
            "callbacks": [
                tf.keras.callbacks.EarlyStopping(patience=8 if continue_training else 12, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
            ],
        }
        if len(X_test) > 0:
            fit_kwargs["validation_data"] = (X_test, y_test)

        history = self.model.fit(X_train, y_train, **fit_kwargs)

        train_pred = self.model.predict(X_train, verbose=0).reshape(-1)
        test_pred = self.model.predict(X_test, verbose=0).reshape(-1)

        close_idx = features_df.columns.get_loc("close")
        feature_count = len(features_df.columns)
        train_actual_price = self._inverse_close_values(y_train, feature_count, close_idx)
        test_actual_price = self._inverse_close_values(y_test, feature_count, close_idx)
        train_pred_price = self._inverse_close_values(train_pred, feature_count, close_idx)
        test_pred_price = self._inverse_close_values(test_pred, feature_count, close_idx)

        train_metrics = self._calculate_regression_metrics(train_actual_price, train_pred_price)
        test_metrics = self._calculate_regression_metrics(test_actual_price, test_pred_price)

        prev_close_test = self._inverse_close_values(X_test[:, -1, close_idx], feature_count, close_idx)
        actual_direction = np.sign(test_actual_price - prev_close_test)
        predicted_direction = np.sign(test_pred_price - prev_close_test)
        directional_accuracy = float(np.mean(actual_direction == predicted_direction) * 100.0)

        fine_tune_count = int(previous_metadata.get("fine_tune_count", 0)) + (1 if continue_training else 0)
        baseline_accuracy = previous_metadata.get("monitoring_baseline_accuracy_pct")
        if baseline_accuracy is None:
            baseline_accuracy = test_metrics["accuracy_pct"]

        metadata = {
            "symbol": symbol,
            "interval": interval,
            "architecture": "conv-bilstm-gru",
            "train_rmse": train_metrics["rmse"],
            "test_rmse": test_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "test_mae": test_metrics["mae"],
            "train_mape": train_metrics["mape"],
            "test_mape": test_metrics["mape"],
            "test_accuracy_pct": test_metrics["accuracy_pct"],
            "directional_accuracy_pct": round(directional_accuracy, 6),
            "feature_columns": list(self.feature_columns),
            "sequence_length": sequence_length,
            "training_rows": int(len(features_df)),
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "model_version": pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S"),
            "monitoring_baseline_accuracy_pct": baseline_accuracy,
            "last_monitor_accuracy_pct": previous_metadata.get("last_monitor_accuracy_pct"),
            "last_monitor_streak": int(previous_metadata.get("last_monitor_streak", 0)),
            "fine_tune_count": fine_tune_count,
            "last_fine_tuned_at": pd.Timestamp.utcnow().isoformat() if continue_training else previous_metadata.get("last_fine_tuned_at"),
        }

        self.save_model(symbol, interval=interval, metadata=metadata)

        return {
            **metadata,
            "history": history.history,
        }

    def fine_tune(self, df, symbol, interval: str = "1h"):
        settings = self._interval_settings(interval)
        return self.train(
            df,
            symbol,
            interval=interval,
            continue_training=True,
            recent_window=int(settings["recent_tune_window"]),
            epochs=int(settings["fine_tune_epochs"]),
            learning_rate=float(settings["fine_tune_learning_rate"]),
        )

    def predict(self, recent_data, interval: str = "1h"):
        """Make predictions on recent data."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        sequence_length = self.get_sequence_length(interval)
        features_df = self.prepare_features(recent_data, interval=interval)
        if len(features_df) < sequence_length:
            raise ValueError(
                f"Not enough recent rows to predict {interval}. Need {sequence_length}, found {len(features_df)}."
            )

        scaled_data = self.scaler.transform(features_df)
        last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, -1)
        prediction = self.model.predict(last_sequence, verbose=0)

        close_col_idx = features_df.columns.get_loc("close")
        predicted_price = self._inverse_close_values(prediction.reshape(-1), len(features_df.columns), close_col_idx)[0]
        return float(predicted_price)

    def save_metadata(self, symbol, interval: str, metadata: dict):
        model_dir = self.model_dir(symbol, interval)
        os.makedirs(model_dir, exist_ok=True)
        metadata_path = f"{model_dir}/metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def save_model(self, symbol, interval: str = "1h", metadata=None):
        """Save model and scaler."""
        model_dir = self.model_dir(symbol, interval)
        os.makedirs(model_dir, exist_ok=True)
        scaler_path = f"{model_dir}/scaler.pkl"
        scaler_hash_path = f"{model_dir}/scaler.pkl.sha256"

        self.model.save(self.model_path(symbol, interval))
        joblib.dump(self.scaler, scaler_path)

        with open(scaler_path, "rb") as f:
            scaler_hash = hashlib.sha256(f.read()).hexdigest()
        with open(scaler_hash_path, "w", encoding="utf-8") as f:
            f.write(scaler_hash)

        if metadata is not None:
            self.save_metadata(symbol, interval, metadata)

    def load_model(self, symbol, interval: str = "1h"):
        """Load saved model and scaler."""
        model_dir = self.model_dir(symbol, interval)
        scaler_path = f"{model_dir}/scaler.pkl"
        scaler_hash_path = f"{model_dir}/scaler.pkl.sha256"
        model_path = self.model_path(symbol, interval)

        if not os.path.exists(model_path) and interval == "1h":
            legacy_dir = self._legacy_model_dir(symbol)
            model_dir = legacy_dir
            scaler_path = f"{model_dir}/scaler.pkl"
            scaler_hash_path = f"{model_dir}/scaler.pkl.sha256"
            model_path = f"{model_dir}/lstm_model.h5"

        self.model = tf.keras.models.load_model(model_path)

        if not os.path.exists(scaler_hash_path):
            raise FileNotFoundError(
                f"Missing scaler integrity file: {scaler_hash_path}. Refuse to load unsigned scaler artifact."
            )

        with open(scaler_hash_path, "r", encoding="utf-8") as f:
            expected_hash = f.read().strip()
        with open(scaler_path, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()

        if actual_hash != expected_hash:
            raise ValueError("Scaler artifact integrity check failed.")

        self.scaler = joblib.load(scaler_path)
        return self.model

    def load_metadata(self, symbol, interval: str = "1h"):
        """Load saved training metadata if available."""
        metadata_path = f"{self.model_dir(symbol, interval)}/metadata.json"
        if not os.path.exists(metadata_path) and interval == "1h":
            metadata_path = f"{self._legacy_model_dir(symbol)}/metadata.json"
        if not os.path.exists(metadata_path):
            return {}
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
