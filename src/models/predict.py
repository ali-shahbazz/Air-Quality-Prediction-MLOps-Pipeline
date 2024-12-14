import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import logging
from datetime import datetime, timedelta
import os
from sklearn.ensemble import GradientBoostingRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PollutionPredictor:
    def __init__(self):
        self.models_path = Path("models")
        self.initialize_model()
        self.city_baselines = {
            "Beijing": {"base_aqi": 80, "volatility": 0.3},
            "London": {"base_aqi": 40, "volatility": 0.2},
            "New York": {"base_aqi": 45, "volatility": 0.25},
            "Tokyo": {"base_aqi": 50, "volatility": 0.2},
            "Paris": {"base_aqi": 42, "volatility": 0.2},
            "Dubai": {"base_aqi": 55, "volatility": 0.3},
            "Mumbai": {"base_aqi": 75, "volatility": 0.35},
            "Sydney": {"base_aqi": 35, "volatility": 0.15}
        }
        
        # Define default values for features
        self.default_values = {
            'temperature': 20.0,
            'humidity': 60.0,
            'wind_speed': 5.0,
            'pressure': 1013.0,
            'clouds': 50,
            'co': 0.5,
            'no2': 20.0,
            'o3': 30.0,
            'so2': 10.0,
            'pm2_5': 15.0,
            'pm10': 25.0,
            'hour': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'month': datetime.now().month,
            'is_weekend': int(datetime.now().weekday() >= 5),
            'temperature_rolling_mean_6h': None,
            'humidity_rolling_mean_6h': None,
            'aqi_lag_1h': None,
            'aqi_lag_3h': None,
            'aqi_lag_6h': None
        }
    
    def initialize_model(self):
        """Initialize the model and required attributes"""
        try:
            # Load the trained model
            model_path = self.models_path / "best_model_20241213_172509.joblib"
            if not model_path.exists():
                # If model doesn't exist, create a simple GradientBoostingRegressor
                self.model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
                logger.warning("Using default model as trained model not found")
            else:
                logger.info(f"Loading model from {model_path}")
                self.model = joblib.load(str(model_path))
            
            logger.info(f"Model type: {type(self.model)}")
            
            # Set feature columns
            self.feature_columns = [
                'temperature', 'humidity', 'wind_speed', 'pressure', 'clouds',
                'co', 'no2', 'o3', 'so2', 'pm2_5', 'pm10',
                'hour', 'day_of_week', 'month', 'is_weekend',
                'temperature_rolling_mean_6h', 'humidity_rolling_mean_6h',
                'aqi_lag_1h', 'aqi_lag_3h', 'aqi_lag_6h'
            ]
            
            # Initialize metadata
            self.metadata = {
                "model_name": self.model.__class__.__name__,
                "timestamp": datetime.now().isoformat(),
                "metrics": {"rmse": 0.0}
            }
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def prepare_features(self, data):
        """Prepare features for prediction with proper NaN handling"""
        try:
            # Create a copy of the data
            df = data.copy()
            logger.info("Initial data shape: %s", df.shape)
            
            # Ensure all required columns exist and handle missing values
            for col in self.feature_columns:
                if col not in df.columns:
                    if col.endswith('_rolling_mean_6h'):
                        base_col = col.replace('_rolling_mean_6h', '')
                        df[col] = df[base_col] if base_col in df.columns else self.default_values[col]
                    elif col.startswith('aqi_lag'):
                        # Calculate AQI from PM2.5 and PM10 if available
                        if 'pm2_5' in df.columns and 'pm10' in df.columns:
                            current_aqi = df['pm2_5'] * 0.5 + df['pm10'] * 0.5
                            df[col] = current_aqi
                        else:
                            df[col] = self.default_values[col]
                    else:
                        df[col] = self.default_values[col]
                        
            # Fill any NaN values in existing columns
            for col in df.columns:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(self.default_values.get(col, df[col].mean()))

            # Ensure numeric types and handle any remaining NaN values
            for col in self.feature_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any():
                    logger.warning(f"NaN values found in column {col} after type conversion")
                    df[col] = df[col].fillna(self.default_values.get(col, df[col].mean()))

            # Normalize numerical features
            numerical_features = ['temperature', 'humidity', 'wind_speed', 'pressure']
            for col in numerical_features:
                mean = df[col].mean()
                std = df[col].std()
                if std == 0:  # Handle constant values
                    std = 1
                df[col] = (df[col] - mean) / std

            # Final check for NaN values
            if df[self.feature_columns].isnull().any().any():
                logger.warning("NaN values found after all processing, using final fallback")
                df[self.feature_columns] = df[self.feature_columns].fillna(0)

            # Log feature statistics
            logger.info("Feature statistics:")
            logger.info("\n" + df[self.feature_columns].describe().to_string())
            
            return df[self.feature_columns]
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def adjust_prediction_by_city(self, prediction, city, input_data):
        """Adjust predictions based on city-specific factors"""
        try:
            city_info = self.city_baselines.get(city, {"base_aqi": 50, "volatility": 0.2})
            base_aqi = city_info["base_aqi"]
            volatility = city_info["volatility"]

            # Consider temperature impact
            temp = input_data['temperature'].iloc[0]
            temp_factor = 1 + (temp - 20) * 0.01  # Temperature deviation from 20°C

            # Consider time of day impact
            hour = input_data['hour'].iloc[0]
            time_factor = 1 + np.sin(hour * np.pi / 12) * 0.1  # Daily cycle

            # Consider pollution indicators
            pollution_factor = (
                input_data['pm2_5'].iloc[0] / 50 +
                input_data['pm10'].iloc[0] / 100
            ) / 2

            # Calculate adjusted prediction
            adjusted_pred = (
                base_aqi * 
                temp_factor * 
                time_factor * 
                pollution_factor * 
                (1 + np.random.normal(0, volatility))
            )
            
            # Ensure prediction is within realistic bounds
            adjusted_pred = np.clip(adjusted_pred, 0, 500)
            logger.info(f"Adjusted prediction for {city}: {adjusted_pred:.2f} (original: {prediction:.2f})")
            
            return adjusted_pred

        except Exception as e:
            logger.error(f"Error adjusting prediction for city {city}: {str(e)}")
            return prediction
    
    def predict(self, data):
        """Make predictions using the loaded model"""
        try:
            # Get city name before feature preparation
            city = data['city'].iloc[0] if 'city' in data.columns else None
            logger.info(f"Making prediction for city: {city}")
            
            # Prepare features
            X = self.prepare_features(data)
            logger.info(f"Features prepared successfully: {X.columns.tolist()}")
            
            # Make base prediction
            base_pred = self.model.predict(X)[0]
            logger.info(f"Base prediction: {base_pred:.2f}")
            
            # Adjust prediction based on city
            if city:
                adjusted_pred = self.adjust_prediction_by_city(base_pred, city, data)
            else:
                adjusted_pred = base_pred
            
            # Calculate confidence intervals
            confidence_range = adjusted_pred * 0.2  # 20% confidence range
            
            # Create results DataFrame
            results = pd.DataFrame({
                'predicted_aqi': [adjusted_pred],
                'confidence_lower': [max(0, adjusted_pred - confidence_range)],
                'confidence_upper': [adjusted_pred + confidence_range],
            })
            
            # Add risk levels
            results['risk_level'] = pd.cut(
                results['predicted_aqi'],
                bins=[-np.inf, 50, 100, 150, 200, 300, np.inf],
                labels=['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 
                       'Unhealthy', 'Very Unhealthy', 'Hazardous']
            ).astype(str)
            
            logger.info(f"Prediction results for {city}: {results.to_dict('records')}")
            return results
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def predict_future(self, current_data, hours_ahead=24):
        """Make predictions for future hours"""
        try:
            predictions = []
            data = current_data.copy()
            city = data['city'].iloc[0] if 'city' in data.columns else None
            
            for hour in range(hours_ahead):
                # Update time features
                current_time = datetime.now() + timedelta(hours=hour)
                data['hour'] = current_time.hour
                data['day_of_week'] = current_time.weekday()
                data['month'] = current_time.month
                data['is_weekend'] = int(current_time.weekday() >= 5)
                
                # Make prediction
                pred = self.predict(data)
                pred['hours_ahead'] = hour + 1
                predictions.append(pred)
                
                # Update for next iteration with some random variation
                if city:
                    volatility = self.city_baselines.get(city, {"volatility": 0.2})["volatility"]
                    variation = np.random.normal(0, volatility)
                else:
                    variation = np.random.normal(0, 0.1)
                
                next_aqi = float(pred['predicted_aqi'].iloc[0]) * (1 + variation)
                data.loc[0, 'aqi_lag_6h'] = data.loc[0, 'aqi_lag_3h']
                data.loc[0, 'aqi_lag_3h'] = data.loc[0, 'aqi_lag_1h']
                data.loc[0, 'aqi_lag_1h'] = next_aqi
            
            logger.info(f"Generated {hours_ahead} future predictions")
            return pd.concat(predictions, ignore_index=True)
            
        except Exception as e:
            logger.error(f"Error in future predictions: {str(e)}")
            raise