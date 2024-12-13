# src/data/preprocessor.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import logging
import json
import joblib
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.raw_data_path = Path("data/raw")
        self.processed_data_path = Path("data/processed")
        self.scaler = StandardScaler()
        self.numeric_columns = [
            'temperature', 'humidity', 'wind_speed', 'pressure', 'clouds',
            'aqi', 'co', 'no2', 'o3', 'so2', 'pm2_5', 'pm10'
        ]
        
    def load_raw_data(self):
        """Load all CSV files from raw data directory"""
        all_files = list(self.raw_data_path.glob("*.csv"))
        if not all_files:
            raise ValueError("No CSV files found in raw data directory")
            
        df_list = []
        for file in all_files:
            try:
                df = pd.read_csv(file)
                df_list.append(df)
                logger.info(f"Loaded file: {file}")
            except Exception as e:
                logger.error(f"Error reading file {file}: {str(e)}")
                
        df = pd.concat(df_list, axis=0, ignore_index=True)
        logger.info(f"Total records loaded: {len(df)}")
        return df
    
    def clean_data(self, df):
        """Clean and prepare data for modeling"""
        logger.info("Starting data cleaning process...")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Handle missing values
        before_cleaning = len(df)
        
        # Forward fill then backward fill missing values
        df[self.numeric_columns] = df[self.numeric_columns].fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers using IQR method
        for col in self.numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[~((df[col] < lower_bound) | (df[col] > upper_bound))]
        
        after_cleaning = len(df)
        logger.info(f"Removed {before_cleaning - after_cleaning} records during cleaning")
        
        return df
    
    def add_features(self, df):
        """Add new features for modeling"""
        logger.info("Adding new features...")
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['season'] = df['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })
        
        # Rolling averages for each city
        for col in self.numeric_columns:
            # 6-hour rolling average
            df[f'{col}_rolling_mean_6h'] = df.groupby('city')[col].transform(
                lambda x: x.rolling(window=6, min_periods=1).mean()
            )
            
            # 24-hour rolling average
            df[f'{col}_rolling_mean_24h'] = df.groupby('city')[col].transform(
                lambda x: x.rolling(window=24, min_periods=1).mean()
            )
        
        # Lag features for AQI
        df['aqi_lag_1h'] = df.groupby('city')['aqi'].shift(1)
        df['aqi_lag_3h'] = df.groupby('city')['aqi'].shift(3)
        df['aqi_lag_6h'] = df.groupby('city')['aqi'].shift(6)
        
        # Fill NaN values created by shifts with forward fill
        df = df.fillna(method='ffill')
        
        logger.info(f"Added features. New shape: {df.shape}")
        return df
    
    def normalize_features(self, df):
        """Normalize numerical features"""
        logger.info("Normalizing features...")
        
        # Get all numeric columns including the new rolling