import numpy as np
import pandas as pd
import tensorflow as tf
import hashlib
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

class LSTMPredictor:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns = config['MODEL_CONFIG']['features']
        self.sequence_length = config['MODEL_CONFIG']['sequence_length']
        
    def prepare_features(self, df):
        """Prepare technical indicators and features"""
        # Sort by timestamp
        df = df.sort_values('timestamp').copy()
        
        # Calculate technical indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        
        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD calculation
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        sma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = sma + (std * 2)
        df['bollinger_lower'] = sma - (std * 2)
        
        # Fill NaN values
        df = df.bfill().ffill()
        
        return df[self.feature_columns]
    
    def create_sequences(self, data, target_col='close'):
        """Create sequences for LSTM training"""
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

        for i in range(self.sequence_length, len(values)):
            X.append(values[i-self.sequence_length:i])
            y.append(values[i, target_idx])

        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2), 
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config['MODEL_CONFIG']['learning_rate']),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    def train(self, df, symbol):
        """Train the LSTM model"""
        # Prepare features
        features_df = self.prepare_features(df)
        minimum_rows = self.sequence_length + 24
        if len(features_df) < minimum_rows:
            raise ValueError(
                f"Not enough rows to train {symbol}. Need at least {minimum_rows}, found {len(features_df)}."
            )
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(features_df)
        scaled_df = pd.DataFrame(scaled_data, columns=features_df.columns)
        
        # Create sequences
        X, y = self.create_sequences(scaled_df)
        if len(X) < 10:
            raise ValueError(
                f"Not enough training sequences for {symbol}. Need at least 10, found {len(X)}."
            )
        
        # Train-test split
        split_idx = int(len(X) * self.config['MODEL_CONFIG']['train_test_split'])
        split_idx = min(max(split_idx, 1), len(X) - 1)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build and train model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config['MODEL_CONFIG']['batch_size'],
            epochs=self.config['MODEL_CONFIG']['epochs'],
            validation_data=(X_test, y_test),
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )

        # Evaluate model
        train_pred = self.model.predict(X_train, verbose=0)
        test_pred = self.model.predict(X_test, verbose=0)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        metadata = {
            'symbol': symbol,
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'feature_columns': list(self.feature_columns),
            'sequence_length': int(self.sequence_length),
            'training_rows': int(len(features_df)),
            'trained_at': pd.Timestamp.utcnow().isoformat(),
            'model_version': pd.Timestamp.utcnow().strftime('%Y%m%d%H%M%S'),
        }

        # Save model, scaler, and metadata
        self.save_model(symbol, metadata)

        return {
            **metadata,
            'history': history.history
        }
    
    def predict(self, recent_data):
        """Make predictions on recent data"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
            
        # Prepare features
        features_df = self.prepare_features(recent_data)
        
        # Scale data
        scaled_data = self.scaler.transform(features_df)
        # Get last sequence
        last_sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        # Predict
        prediction = self.model.predict(last_sequence, verbose=0)
        
        # Inverse scale prediction (only for close price)
        close_col_idx = features_df.columns.get_loc('close')
        dummy_array = np.zeros((1, len(features_df.columns)))
        dummy_array[0, close_col_idx] = prediction[0, 0]
        
        inverse_scaled = self.scaler.inverse_transform(dummy_array)
        predicted_price = inverse_scaled[0, close_col_idx]
        
        return predicted_price
    
    def save_model(self, symbol, metadata=None):
        """Save model and scaler"""
        model_dir = f"models/saved/{symbol}"
        os.makedirs(model_dir, exist_ok=True)
        scaler_path = f"{model_dir}/scaler.pkl"
        scaler_hash_path = f"{model_dir}/scaler.pkl.sha256"
        metadata_path = f"{model_dir}/metadata.json"
        
        self.model.save(f"{model_dir}/lstm_model.h5")
        joblib.dump(self.scaler, scaler_path)

        with open(scaler_path, "rb") as f:
            scaler_hash = hashlib.sha256(f.read()).hexdigest()
        with open(scaler_hash_path, "w", encoding="utf-8") as f:
            f.write(scaler_hash)

        if metadata is not None:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        
    def load_model(self, symbol):
        """Load saved model and scaler"""
        model_dir = f"models/saved/{symbol}"
        scaler_path = f"{model_dir}/scaler.pkl"
        scaler_hash_path = f"{model_dir}/scaler.pkl.sha256"
        
        self.model = tf.keras.models.load_model(f"{model_dir}/lstm_model.h5")

        if not os.path.exists(scaler_hash_path):
            raise FileNotFoundError(
                f"Missing scaler integrity file: {scaler_hash_path}. "
                "Refuse to load unsigned scaler artifact."
            )

        with open(scaler_hash_path, "r", encoding="utf-8") as f:
            expected_hash = f.read().strip()
        with open(scaler_path, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()

        if actual_hash != expected_hash:
            raise ValueError("Scaler artifact integrity check failed.")

        self.scaler = joblib.load(scaler_path)

    def load_metadata(self, symbol):
        """Load saved training metadata if available."""
        metadata_path = f"models/saved/{symbol}/metadata.json"
        if not os.path.exists(metadata_path):
            return {}
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
