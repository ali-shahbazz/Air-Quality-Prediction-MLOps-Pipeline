# src/models/train.py

import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import json
from datetime import datetime
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.data_path = Path("data/processed/processed_data.csv")
        self.models_path = Path("models")
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Set up MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("pollution_prediction")
        
        # Define feature columns
        self.feature_columns = [
            'temperature', 'humidity', 'wind_speed', 'pressure', 'clouds',
            'co', 'no2', 'o3', 'so2', 'pm2_5', 'pm10',
            'hour', 'day_of_week', 'month', 'is_weekend',
            'temperature_rolling_mean_6h', 'humidity_rolling_mean_6h',
            'aqi_lag_1h', 'aqi_lag_3h', 'aqi_lag_6h'
        ]
        self.target_column = 'aqi'

    def prepare_data(self):
        """Load and prepare data for training"""
        logger.info("Loading processed data...")
        df = pd.read_csv(self.data_path)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Split features and target
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        # Perform time-series split
        tscv = TimeSeriesSplit(n_splits=5)
        train_index, test_index = list(tscv.split(X))[-1]
        
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        
        return X_train, X_test, y_train, y_test

    def train_models(self):
        """Train models with hyperparameter tuning"""
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        models = {
            'RandomForest': RandomForestRegressor(),
            'GradientBoosting': GradientBoostingRegressor()
        }
        
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'GradientBoosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
        
        best_model = None
        best_rmse = float('inf')
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name} with hyperparameter tuning...")
            
            grid_search = GridSearchCV(model, param_grids[model_name], cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            logger.info(f"Best parameters for {model_name}: {best_params}")
            
            # Evaluate the best model
            y_pred = best_model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Log metrics and model with MLflow
            with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"):
                mlflow.log_params(best_params)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
                mlflow.sklearn.log_model(best_model, model_name)
                
                # Save feature importance plot
                if hasattr(best_model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': self.feature_columns,
                        'importance': best_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    feature_importance.to_csv(f"feature_importance_{model_name}.csv")
                    mlflow.log_artifact(f"feature_importance_{model_name}.csv")
                
                logger.info(f"{model_name} metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
                
                # Track best model
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = best_model
        
        # Save the best model
        best_model_path = self.models_path / f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(best_model, best_model_path)
        logger.info(f"Best model saved to {best_model_path}")

def main():
    trainer = ModelTrainer()
    trainer.train_models()

if __name__ == "__main__":
    main()