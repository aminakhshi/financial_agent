import numpy as np
import pandas as pd
import tensorflow as tf
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
        df = df.sort_values('timestamp')
        
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
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df[self.feature_columns]
    
    def create_sequences(self, data, target_col='close'):
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, data.columns.get_loc(target_col)])
            
        return np.array(X), np.array(y)
    
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
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(features_df)
        scaled_df = pd.DataFrame(scaled_data, columns=features_df.columns)
        
        # Create sequences
        X, y = self.create_sequences(scaled_df)
        
        # Train-test split
        split_idx = int(len(X) * self.config['MODEL_CONFIG']['train_test_split'])
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build and train model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.config['MODEL_CONFIG']['batch_size'],
            epochs=self.config['MODEL_CONFIG']['epochs'],
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Testing RMSE: {test_rmse:.4f}")
        
        # Save model and scaler
        self.save_model(symbol)
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
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
        scaled_df = pd.DataFrame(scaled_data, columns=features_df.columns)
        
        # Get last sequence
        last_sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        # Predict
        prediction = self.model.predict(last_sequence)
        
        # Inverse scale prediction (only for close price)
        close_col_idx = features_df.columns.get_loc('close')
        dummy_array = np.zeros((1, len(features_df.columns)))
        dummy_array[0, close_col_idx] = prediction[0, 0]
        
        inverse_scaled = self.scaler.inverse_transform(dummy_array)
        predicted_price = inverse_scaled[0, close_col_idx]
        
        return predicted_price
    
    def save_model(self, symbol):
        """Save model and scaler"""
        model_dir = f"models/saved/{symbol}"
        os.makedirs(model_dir, exist_ok=True)
        
        self.model.save(f"{model_dir}/lstm_model.h5")
        joblib.dump(self.scaler, f"{model_dir}/scaler.pkl")
        
    def load_model(self, symbol):
        """Load saved model and scaler"""
        model_dir = f"models/saved/{symbol}"
        
        self.model = tf.keras.models.load_model(f"{model_dir}/lstm_model.h5")
        self.scaler = joblib.load(f"{model_dir}/scaler.pkl")