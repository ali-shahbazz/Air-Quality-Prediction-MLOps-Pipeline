# src/models/predict.py

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PollutionPredictor:
    def __init__(self):
        self.models_path = Path("models")
        self.load_model()
        
    def load_model(self):
        """Load the latest trained model"""
        try:
            # Find the latest model file
            model_files = list(self.models_path.glob("best_model_*.joblib"))
            if not model_files:
                raise FileNotFoundError("No trained model found")
            
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            self.model = joblib.load(latest_model)
            
            # Load model metadata
            with open(self.models_path / "model_metadata.json", 'r') as f:
                self.metadata = json.load(f)
            
            logger.info(f"Loaded model: {latest_model}")
            self.feature_columns = self.metadata['metrics']['feature_columns']
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def prepare_features(self, data):
        """Prepare features for prediction"""
        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        return data[self.feature_columns]
    
    def predict(self, data):
        """Make predictions using the loaded model"""
        try:
            # Prepare features
            X = self.prepare_features(data)
            
            # Make predictions
            predictions = self.model.predict(X)
            
            # Add confidence intervals if using Random Forest
            if hasattr(self.model, 'estimators_'):
                predictions_all = np.array([tree.predict(X) for tree in self.model.estimators_])
                confidence_lower = np.percentile(predictions_all, 2.5, axis=0)
                confidence_upper = np.percentile(predictions_all, 97.5, axis=0)
            else:
                confidence_lower = predictions - predictions * 0.1  # Simplified confidence interval
                confidence_upper = predictions + predictions * 0.1
            
            # Create results DataFrame
            results = pd.DataFrame({
                'predicted_aqi': predictions,
                'confidence_lower': confidence_lower,
                'confidence_upper': confidence_upper
            })
            
            # Add risk levels
            results['risk_level'] = pd.cut(
                results['predicted_aqi'],
                bins=[0, 50, 100, 150, 200, 300, float('inf')],
                labels=['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 
                       'Unhealthy', 'Very Unhealthy', 'Hazardous']
            )
            
            return results
        
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_future(self, current_data, hours_ahead=24):
        """Make predictions for future hours"""
        predictions = []
        current_data = current_data.copy()
        
        for hour in range(hours_ahead):
            # Make prediction for current hour
            pred = self.predict(current_data)
            pred['hours_ahead'] = hour + 1
            predictions.append(pred)
            
            # Update features for next prediction
            current_data['aqi_lag_6h'] = current_data['aqi_lag_3h']
            current_data['aqi_lag_3h'] = current_data['aqi_lag_1h']
            current_data['aqi_lag_1h'] = pred['predicted_aqi'].values[0]
            
            # Update time-based features
            current_time = datetime.now() + timedelta(hours=hour+1)
            current_data['hour'] = current_time.hour
            current_data['day_of_week'] = current_time.weekday()
            current_data['is_weekend'] = int(current_time.weekday() >= 5)
        
        return pd.concat(predictions)

def main():
    try:
        # Example usage
        predictor = PollutionPredictor()
        
        # Load some test data
        test_data = pd.read_csv("data/processed/processed_data.csv").iloc[-1:]
        
        # Make predictions
        predictions = predictor.predict(test_data)
        logger.info(f"Predictions:\n{predictions}")
        
        # Make future predictions
        future_predictions = predictor.predict_future(test_data)
        logger.info(f"Future predictions:\n{future_predictions}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()