# src/api/app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import sys
from pathlib import Path
import logging
from datetime import datetime
import json

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.predict import PollutionPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Environmental Monitoring API",
    description="API for environmental data monitoring and pollution prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = PollutionPredictor()

# Pydantic models for request/response
class PredictionInput(BaseModel):
    city: str
    temperature: float
    humidity: float
    wind_speed: float
    pressure: float
    clouds: int
    co: float
    no2: float
    o3: float
    so2: float
    pm2_5: float
    pm10: float
    aqi_lag_1h: Optional[float] = None
    aqi_lag_3h: Optional[float] = None
    aqi_lag_6h: Optional[float] = None

class PredictionResponse(BaseModel):
    predicted_aqi: float
    confidence_lower: float
    confidence_upper: float
    risk_level: str
    timestamp: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Environmental Monitoring API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Verify model is loaded
        if predictor.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return {
            "status": "healthy",
            "model_info": {
                "type": predictor.metadata["model_name"],
                "last_trained": predictor.metadata["timestamp"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str)# src/api/app.py (continued)

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    """Make single prediction"""
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame([input_data.dict()])
        
        # Add time-based features
        current_time = datetime.now()
        data['hour'] = current_time.hour
        data['day_of_week'] = current_time.weekday()
        data['month'] = current_time.month
        data['is_weekend'] = int(current_time.weekday() >= 5)
        
        # Fill missing lag values if not provided
        if input_data.aqi_lag_1h is None:
            data['aqi_lag_1h'] = data['pm2_5'] * 0.5  # Approximate
        if input_data.aqi_lag_3h is None:
            data['aqi_lag_3h'] = data['aqi_lag_1h']
        if input_data.aqi_lag_6h is None:
            data['aqi_lag_6h'] = data['aqi_lag_3h']
        
        # Make prediction
        predictions = predictor.predict(data)
        
        # Prepare response
        response = {
            "predicted_aqi": float(predictions['predicted_aqi'].iloc[0]),
            "confidence_lower": float(predictions['confidence_lower'].iloc[0]),
            "confidence_upper": float(predictions['confidence_upper'].iloc[0]),
            "risk_level": predictions['risk_level'].iloc[0],
            "timestamp": current_time.isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/future")
async def predict_future(input_data: PredictionInput, hours_ahead: int = 24):
    """Make predictions for future hours"""
    try:
        if hours_ahead > 72:
            raise HTTPException(
                status_code=400,
                detail="Cannot predict more than 72 hours ahead"
            )
        
        # Convert input data to DataFrame
        data = pd.DataFrame([input_data.dict()])
        
        # Add time-based features
        current_time = datetime.now()
        data['hour'] = current_time.hour
        data['day_of_week'] = current_time.weekday()
        data['month'] = current_time.month
        data['is_weekend'] = int(current_time.weekday() >= 5)
        
        # Make future predictions
        future_predictions = predictor.predict_future(data, hours_ahead)
        
        # Prepare response
        response = []
        for idx, row in future_predictions.iterrows():
            prediction = {
                "hours_ahead": int(row['hours_ahead']),
                "predicted_aqi": float(row['predicted_aqi']),
                "confidence_lower": float(row['confidence_lower']),
                "confidence_upper": float(row['confidence_upper']),
                "risk_level": row['risk_level'],
                "timestamp": (current_time + pd.Timedelta(hours=int(row['hours_ahead']))).isoformat()
            }
            response.append(prediction)
        
        return response
        
    except Exception as e:
        logger.error(f"Future prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cities")
async def get_cities():
    """Get list of supported cities"""
    return {
        "cities": [
            {"name": "New York", "lat": 40.7128, "lon": -74.0060},
            {"name": "London", "lat": 51.5074, "lon": -0.1278},
            {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
            {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
            {"name": "Beijing", "lat": 39.9042, "lon": 116.4074},
            {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
            {"name": "Dubai", "lat": 25.2048, "lon": 55.2708},
            {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777}
        ]
    }

@app.get("/model/info")
async def get_model_info():
    """Get information about the current model"""
    return {
        "model_type": predictor.metadata["model_name"],
        "last_trained": predictor.metadata["timestamp"],
        "metrics": predictor.metadata["metrics"],
        "features": predictor.feature_columns
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)