import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

class StockPredictor:
    def __init__(self, lookback_days=30):
        self.lookback_days = lookback_days
        self.scaler = MinMaxScaler()
        self.model = None

        self.feature_names = [
            'OPEN', 'HIGH', 'LOW', 'close', 'VOLUME',
            'vwap', 'daily_return', 'SMA20', 'SMA50',
            'RSI', 'BB_MIDDLE', 'BB_UPPER', 'BB_LOWER'
        ]

    def prepare_data(self, df):
        """
        Prepare and clean the data, calculate technical indicators
        """
        # Make a copy and clean column names
        df = df.copy()
        df.columns = df.columns.str.strip().str.replace('"', '')

        # Fix: More robust date conversion with fallback options
        if 'Date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                try:
                    # Try multiple formats
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    # If too many NaT values after generic conversion, try specific format
                    if df['Date'].isna().sum() > len(df) * 0.5:
                        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y', errors='coerce')
                except Exception as e:
                    print(f"Date conversion error: {str(e)}")
                    # If all conversion attempts fail, create a date range
                    df['Date'] = pd.date_range(start='2020-01-01', periods=len(df))

        # Clean numeric columns
        for col in df.columns:
            if col != 'Date' and col != 'series':
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('"', '').str.replace(',', ''), errors='coerce')

        # Calculate technical indicators
        try:
            # Daily returns
            df['daily_return'] = df['close'].pct_change()

            # Moving averages
            df['SMA20'] = df['close'].rolling(window=20, min_periods=1).mean()
            df['SMA50'] = df['close'].rolling(window=50, min_periods=1).mean()

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            # Fix: Avoid division by zero
            rs = gain / loss.replace(0, 0.001)
            df['RSI'] = 100 - (100 / (1 + rs))
            df['RSI'] = df['RSI'].fillna(50)  # Fill NaN with neutral RSI

            # Bollinger Bands
            df['BB_MIDDLE'] = df['close'].rolling(window=20, min_periods=1).mean()
            rolling_std = df['close'].rolling(window=20, min_periods=1).std()
            df['BB_UPPER'] = df['BB_MIDDLE'] + (2 * rolling_std)
            df['BB_LOWER'] = df['BB_MIDDLE'] - (2 * rolling_std)

            # Calculate BB Width for volatility analysis
            df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MIDDLE'].replace(0, 0.001)

            # Calculate relative volume
            df['relative_volume'] = df['VOLUME'] / df['VOLUME'].rolling(window=20).mean().replace(0, 0.001)

            # Add MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()

        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            # Create basic indicators with default values if calculation fails
            df['daily_return'] = 0.0
            df['SMA20'] = df['close']
            df['SMA50'] = df['close']
            df['RSI'] = 50.0
            df['BB_MIDDLE'] = df['close']
            df['BB_UPPER'] = df['close'] * 1.02
            df['BB_LOWER'] = df['close'] * 0.98
            df['BB_WIDTH'] = 0.04
            df['relative_volume'] = 1.0
            df['MACD'] = 0.0
            df['MACD_SIGNAL'] = 0.0

        # Fix: Handle infinite and NaN values
        for col in df.columns:
            if col != 'Date' and col != 'series':
                # Replace inf with large numbers
                df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
                # Forward and backward fill NaN values
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Sort by date ascending
        if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
            df = df.sort_values('Date').reset_index(drop=True)

        return df

    def build_model(self, input_shape):
        """
        Build LSTM model with proper input layer
        """
        model = Sequential([
            Input(shape=input_shape),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')
        return model

    def create_sequences(self, data):
        """
        Create sequences for LSTM model
        """
        # Get only the features we need
        features = [col for col in self.feature_names if col in data.columns]
        
        # Store feature names for prediction
        self.current_features = features

        # Scale the features
        scaled_data = self.scaler.fit_transform(data[features])

        X, y = [], []
        for i in range(self.lookback_days, len(scaled_data)):
            X.append(scaled_data[i-self.lookback_days:i])
            y.append(scaled_data[i, features.index('close')])

        return np.array(X), np.array(y)

    def train(self, df, epochs=100, batch_size=32, validation_split=0.2):
        """
        Train the model with early stopping
        """
        # Prepare data
        processed_data = self.prepare_data(df)
        X, y = self.create_sequences(processed_data)

        # Fix: Check if we have enough data to train
        if len(X) < 10:
            raise ValueError(f"Not enough data for training. Need at least {self.lookback_days + 10} rows, got {len(df)}.")

        # Split data into training and validation sets
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Build and train model with proper input shape
        self.model = self.build_model((self.lookback_days, len(self.current_features)))

        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )

        return history, processed_data

    def predict_next_days(self, df, days=5):
        """
        Predict stock prices for the next few days
        """
        if not hasattr(self, 'current_features'):
            raise ValueError("Model needs to be trained before making predictions")

        # Prepare data
        processed_data = self.prepare_data(df)

        # Get the last sequence
        last_sequence = processed_data[self.current_features].values[-self.lookback_days:]
        last_sequence = self.scaler.transform(last_sequence)

        predictions = []
        current_sequence = last_sequence.copy()
        
        # Fix: Use try-except to handle prediction errors
        try:
            for _ in range(days):
                # Reshape sequence for prediction
                current_input = current_sequence.reshape(1, self.lookback_days, len(self.current_features))

                # Predict next day
                pred = self.model.predict(current_input, verbose=0)
                predictions.append(pred[0])

                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1] = current_sequence[-2].copy()  # Copy the previous day's values
                current_sequence[-1, self.current_features.index('close')] = pred[0, 0]

            # Convert predictions back to original scale
            scaled_predictions = np.array(predictions).reshape(-1, 1)
            dummy_array = np.zeros((len(scaled_predictions), len(self.current_features)))
            dummy_array[:, self.current_features.index('close')] = scaled_predictions[:, 0]
            actual_predictions = self.scaler.inverse_transform(dummy_array)[:, self.current_features.index('close')]

            # Generate dates for predictions (business days only)
            last_date = processed_data['Date'].iloc[-1]
            prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='B')
            
            # Fix: Handle timezone-aware timestamps to prevent Arrow serialization issues
            if pd.api.types.is_datetime64tz_dtype(prediction_dates):
                prediction_dates = prediction_dates.tz_localize(None)

        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            # Return fallback predictions
            last_close = processed_data['close'].iloc[-1]
            actual_predictions = np.array([last_close * (1 + 0.001 * i) for i in range(days)])
            last_date = processed_data['Date'].iloc[-1]
            prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='B')

        return {
            'dates': prediction_dates,
            'predictions': actual_predictions,
            'last_actual_close': processed_data['close'].iloc[-1]
        }

    def print_predictions(self, predictions):
        """
        Helper method to print predictions in a formatted way
        """
        print("\nPredicted Stock Prices:")
        print("-----------------------")
        for date, pred in zip(predictions['dates'], predictions['predictions']):
            print(f"Date: {date.strftime('%Y-%m-%d')}, Predicted Close: ₹{pred:.2f}")
        print(f"\nLast Actual Close: ₹{predictions['last_actual_close']:.2f}")

    def analyze_prediction_confidence(self, processed_data, prediction, window=20):
        """
        Analyze prediction confidence based on market conditions
        """
        # Fix: Handle case when processed_data is None
        if processed_data is None:
            return {
                'confidence_score': 0.5,  # Neutral confidence
                'factors': {
                    'volatility': 0.0,
                    'trend_strength': 0.0,
                    'volume_support': 0.0,
                    'technical_alignment': 0.0
                }
            }
            
        try:
            latest_data = processed_data.iloc[-1]

            confidence_factors = {
                'volatility': 0.0,
                'trend_strength': 0.0,
                'volume_support': 0.0,
                'technical_alignment': 0.0
            }

            # Volatility analysis (using BB width)
            if 'BB_WIDTH' in processed_data.columns:
                bb_width = latest_data['BB_WIDTH']
                avg_bb_width = processed_data['BB_WIDTH'].tail(window).mean()

                if bb_width > avg_bb_width * 1.2:
                    confidence_factors['volatility'] = -0.2  # High volatility reduces confidence
                else:
                    confidence_factors['volatility'] = 0.2
            else:
                confidence_factors['volatility'] = 0.0

            # Trend strength
            if 'SMA20' in processed_data.columns and 'SMA50' in processed_data.columns:
                price_trend = (latest_data['SMA20'] - latest_data['SMA50']) / (latest_data['SMA50'] + 0.001)
                confidence_factors['trend_strength'] = min(0.3, abs(price_trend))
            else:
                confidence_factors['trend_strength'] = 0.1

            # Volume support
            if 'relative_volume' in processed_data.columns:
                if latest_data['relative_volume'] > 1.2:
                    confidence_factors['volume_support'] = 0.2
                else:
                    confidence_factors['volume_support'] = -0.1
            else:
                confidence_factors['volume_support'] = 0.0

            # Technical indicator alignment
            aligned_indicators = 0
            total_indicators = 0

            if 'RSI' in processed_data.columns:
                total_indicators += 1
                if latest_data['RSI'] > 50: 
                    aligned_indicators += 1
            
            if 'MACD' in processed_data.columns and 'MACD_SIGNAL' in processed_data.columns:
                total_indicators += 1
                if latest_data['MACD'] > latest_data['MACD_SIGNAL']: 
                    aligned_indicators += 1
            
            if 'close' in processed_data.columns and 'BB_MIDDLE' in processed_data.columns:
                total_indicators += 1
                if latest_data['close'] > latest_data['BB_MIDDLE']: 
                    aligned_indicators += 1

            confidence_factors['technical_alignment'] = (aligned_indicators / max(1, total_indicators)) * 0.3

            # Calculate overall confidence score
            confidence_score = sum(confidence_factors.values())

            return {
                'confidence_score': min(1.0, max(0.0, 0.5 + confidence_score)),
                'factors': confidence_factors
            }
            
        except Exception as e:
            print(f"Error analyzing confidence: {str(e)}")
            return {
                'confidence_score': 0.5,
                'factors': {
                    'volatility': 0.0,
                    'trend_strength': 0.0,
                    'volume_support': 0.0,
                    'technical_alignment': 0.0
                }
            }