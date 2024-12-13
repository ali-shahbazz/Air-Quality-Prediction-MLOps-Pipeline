# src/data/preprocessor.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.raw_data_path = Path("data/raw")
        self.processed_data_path = Path("data/processed")
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.numeric_columns = [
            'temperature', 'humidity', 'wind_speed', 'aqi', 
            'co', 'no2', 'pm2_5', 'pm10'
        ]
        
    def load_raw_data(self):
        """Load all CSV files from raw data directory"""
        logger.info("Loading raw data files...")
        all_files = list(self.raw_data_path.glob("environmental_data_*.csv"))
        
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
        logger.info("Cleaning data...")
        
        # Convert timestamp string to proper datetime format
        df['timestamp'] = df['timestamp'].apply(
            lambda x: datetime.strptime(x, "%Y%m%d_%H%M%S")
        )
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Handle missing values
        logger.info(f"Missing values before imputation: {df[self.numeric_columns].isnull().sum().sum()}")
        
        # Fill missing values with mean for each city
        for col in self.numeric_columns:
            city_means = df.groupby('city')[col].transform('mean')
            df[col] = df[col].fillna(city_means)
            # If still any NaN (for cities with all NaN), fill with overall mean
            df[col] = df[col].fillna(df[col].mean())
        
        logger.info(f"Missing values after imputation: {df[self.numeric_columns].isnull().sum().sum()}")
        
        # Remove outliers using IQR method
        before_outliers = len(df)
        for col in self.numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # Using 3 instead of 1.5 to be less aggressive
            upper_bound = Q3 + 3 * IQR
            df = df[~((df[col] < lower_bound) | (df[col] > upper_bound))]
        
        logger.info(f"Removed {before_outliers - len(df)} outliers")
        logger.info(f"Data shape after cleaning: {df.shape}")
        return df

    def add_features(self, df):
        """Add new features for modeling"""
        logger.info("Adding features...")
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Rolling averages for each city
        for col in self.numeric_columns:
            df[f'{col}_rolling_mean_6h'] = df.groupby('city')[col].transform(
                lambda x: x.rolling(window=6, min_periods=1).mean()
            )
            # Fill NaN values with the column mean
            df[f'{col}_rolling_mean_6h'].fillna(df[f'{col}_rolling_mean_6h'].mean(), inplace=True)
        
        # Lag features for AQI with immediate filling
        for city in df['city'].unique():
            city_mask = df['city'] == city
            city_data = df[city_mask].copy()
            
            # Calculate lags and fill with the mean of existing values
            for lag in [1, 3, 6]:
                lag_col = f'aqi_lag_{lag}h'
                df.loc[city_mask, lag_col] = city_data['aqi'].shift(lag)
                df[lag_col].fillna(df['aqi'].mean(), inplace=True)
        
        logger.info(f"Data shape after feature engineering: {df.shape}")
        logger.info(f"Missing values after feature engineering: {df.isnull().sum().sum()}")
        return df
    
    def normalize_features(self, df):
        """Normalize numerical features"""
        logger.info("Normalizing features...")
        
        # Define columns to exclude from normalization
        exclude_cols = ['hour', 'day_of_week', 'month', 'is_weekend', 'city', 'timestamp']
        
        # Get columns to normalize
        normalize_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Fill any remaining NaN values before normalization
        for col in normalize_cols:
            df[col].fillna(df[col].mean(), inplace=True)
        
        # Normalize the features
        df[normalize_cols] = self.scaler.fit_transform(df[normalize_cols])
        
        # Save scaler parameters
        scaler_params = {
            'mean': dict(zip(normalize_cols, self.scaler.mean_)),
            'scale': dict(zip(normalize_cols, self.scaler.scale_))
        }
        
        with open(self.processed_data_path / 'scaler_params.json', 'w') as f:
            json.dump(scaler_params, f, indent=4)
        
        logger.info("Saved scaler parameters")
        logger.info(f"Final missing values check: {df.isnull().sum().sum()}")
        return df
    
    def process_data(self):
        """Main processing pipeline"""
        try:
            logger.info("Starting data processing pipeline...")
            
            # Load raw data
            df = self.load_raw_data()
            
            # Clean data
            df = self.clean_data(df)
            
            # Add features
            df = self.add_features(df)
            
            # Normalize features
            df = self.normalize_features(df)
            
            # Save processed data
            output_file = self.processed_data_path / "processed_data.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Saved processed data to {output_file}")
            
            # Save processing metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "num_records": len(df),
                "num_features": len(df.columns),
                "features": list(df.columns),
                "cities": list(df['city'].unique())
            }
            
            with open(self.processed_data_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info("Data processing completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in data processing pipeline: {str(e)}")
            raise

def main():
    try:
        preprocessor = DataPreprocessor()
        preprocessor.process_data()
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()