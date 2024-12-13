# src/models/predict.py

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
    
    def initialize_model(self):
        """Initialize the model and required attributes"""
        try:
            # Load the trained model
            model_path = self.models_path / "best_model_20241213_163001.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
                
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(str(model_path))
            logger.info(f"Model type: {type(self.model)}")
            
            # Load metadata
            metadata_path = self.models_path / "model_metadata.json"
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info("Loaded metadata successfully")
            
            # Set feature columns
            self.feature_columns = [
                'temperature', 'humidity', 'wind_speed', 'pressure', 'clouds',
                'co', 'no2', 'o3', 'so2', 'pm2_5', 'pm10',
                'hour', 'day_of_week', 'month', 'is_weekend',
                'temperature_rolling_mean_6h', 'humidity_rolling_mean_6h',
                'aqi_lag_1h', 'aqi_lag_3h', 'aqi_lag_6h'
            ]
            
            # Test model predict method
            test_data = pd.DataFrame([[20.5, 65, 5.2, 1013, 75, 
                                     250, 15, 35, 5, 12, 20, 
                                     12, 3, 12, 0, 
                                     21.0, 63.0, 45.0, 42.0, 40.0]], 
                                   columns=self.feature_columns)
            test_pred = self.model.predict(test_data)
            logger.info(f"Test prediction successful: {test_pred}")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def prepare_features(self, data):
        """Prepare features for prediction"""
        try:
            # Select required features
            df = pd.DataFrame(columns=self.feature_columns)
            for col in self.feature_columns:
                if col not in data.columns:
                    raise ValueError(f"Missing required feature: {col}")
                df[col] = data[col].astype(float)
            
            logger.info(f"Prepared features shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def predict(self, data):
        """Make predictions using the loaded model"""
        try:
            # Prepare features
            X = self.prepare_features(data)
            logger.info(f"Features prepared successfully: {X.columns.tolist()}")
            
            # Make prediction
            y_pred = self.model.predict(X)
            logger.info(f"Raw predictions: {y_pred}")
            
            # Calculate confidence intervals
            std = np.std([
                estimator[0].predict(X)
                for estimator in self.model.estimators_
            ], axis=0)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'predicted_aqi': y_pred,
                'confidence_lower': np.maximum(0, y_pred - 1.96 * std),
                'confidence_upper': y_pred + 1.96 * std,
            })
            
            # Add risk levels
            results['risk_level'] = pd.cut(
                results['predicted_aqi'],
                bins=[-np.inf, 50, 100, 150, 200, 300, np.inf],
                labels=['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 
                       'Unhealthy', 'Very Unhealthy', 'Hazardous']
            ).astype(str)
            
            logger.info(f"Prediction results: {results.to_dict('records')}")
            return results
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def predict_future(self, current_data, hours_ahead=24):
        """Make predictions for future hours"""
        try:
            predictions = []
            data = current_data.copy()
            
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
                
                # Update for next iteration
                data.loc[0, 'aqi_lag_6h'] = data.loc[0, 'aqi_lag_3h']
                data.loc[0, 'aqi_lag_3h'] = data.loc[0, 'aqi_lag_1h']
                data.loc[0, 'aqi_lag_1h'] = float(pred['predicted_aqi'].iloc[0])
            
            return pd.concat(predictions, ignore_index=True)
            
        except Exception as e:
            logger.error(f"Error in future predictions: {str(e)}")
            raise