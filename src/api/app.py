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
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app, Summary
import time

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

# Enhanced Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'Request duration',
    ['method', 'endpoint']
)

PREDICTION_ERROR = Gauge(
    'prediction_error',
    'Absolute difference between predicted and actual AQI',
    ['city']
)

PREDICTED_AQI = Gauge(
    'predicted_aqi',
    'Predicted AQI value',
    ['city']
)

MODEL_PREDICTION_LATENCY = Histogram(
    'model_prediction_duration_seconds',
    'Time taken for model prediction',
    ['city'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5]
)

DATA_PROCESSING_LATENCY = Histogram(
    'data_processing_duration_seconds',
    'Time taken for data processing',
    ['city']
)

DATA_INGESTION_RATE = Counter(
    'data_ingestion_total',
    'Total number of data points ingested',
    ['city']
)

API_HEALTH = Gauge(
    'api_health',
    'API health status',
    ['endpoint']
)

FEATURE_STATS = Gauge(
    'feature_stats',
    'Feature statistics',
    ['city', 'feature', 'stat_type']
)

MODEL_METRICS = Gauge(
    'model_metrics',
    'Model performance metrics',
    ['metric_name']
)

DATA_QUALITY = Gauge(
    'data_quality',
    'Data quality metrics',
    ['city', 'metric_type']
)

ERROR_RATE = Counter(
    'prediction_errors_total',
    'Total number of prediction errors',
    ['city', 'error_type']
)

# Add metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
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
    temperature_rolling_mean_6h: Optional[float] = None
    humidity_rolling_mean_6h: Optional[float] = None
    aqi_lag_1h: Optional[float] = None
    aqi_lag_3h: Optional[float] = None
    aqi_lag_6h: Optional[float] = None

class PredictionResponse(BaseModel):
    predicted_aqi: float
    confidence_lower: float
    confidence_upper: float
    risk_level: str
    timestamp: str

# Initialize predictor
try:
    from src.models.predict import PollutionPredictor
    predictor = PollutionPredictor()
    logger.info("Model loaded successfully")
    API_HEALTH.labels(endpoint="/model").set(1)
except Exception as e:
    logger.error(f"Failed to initialize predictor: {str(e)}")
    predictor = None
    API_HEALTH.labels(endpoint="/model").set(0)

def calculate_aqi(pm25, pm10):
    """Calculate AQI from PM2.5 and PM10 values"""
    return (pm25 * 0.5 + pm10 * 0.5) * 2

def check_data_quality(data, city):
    """Check data quality and update metrics"""
    with DATA_PROCESSING_LATENCY.labels(city=city).time():
        missing_values = data.isnull().sum().sum()
        DATA_QUALITY.labels(city=city, metric_type="missing_values").set(missing_values)
        
        # Check for outliers in numerical columns
        for col in ['temperature', 'humidity', 'wind_speed']:
            if col in data.columns:
                mean = data[col].mean()
                std = data[col].std()
                outliers = data[data[col].abs() > mean + 3*std].shape[0]
                DATA_QUALITY.labels(city=city, metric_type=f"{col}_outliers").set(outliers)
                
        # Update data ingestion counter
        DATA_INGESTION_RATE.labels(city=city).inc()

@app.get("/")
async def root():
    """Root endpoint"""
    start_time = time.time()
    try:
        API_HEALTH.labels(endpoint="/").set(1)
        response = {
            "message": "Environmental Monitoring API",
            "version": "1.0.0",
            "status": "active"
        }
        REQUEST_COUNT.labels(method='GET', endpoint='/', status=200).inc()
        REQUEST_LATENCY.labels(method='GET', endpoint='/').observe(time.time() - start_time)
        return response
    except Exception as e:
        API_HEALTH.labels(endpoint="/").set(0)
        REQUEST_COUNT.labels(method='GET', endpoint='/', status=500).inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    start_time = time.time()
    try:
        if predictor is None:
            API_HEALTH.labels(endpoint="/health").set(0)
            REQUEST_COUNT.labels(method='GET', endpoint='/health', status=503).inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        API_HEALTH.labels(endpoint="/health").set(1)
        response = {
            "status": "healthy",
            "model_info": {
                "type": predictor.metadata["model_name"],
                "last_trained": predictor.metadata["timestamp"]
            }
        }
        REQUEST_COUNT.labels(method='GET', endpoint='/health', status=200).inc()
        REQUEST_LATENCY.labels(method='GET', endpoint='/health').observe(time.time() - start_time)
        return response
    except Exception as e:
        REQUEST_COUNT.labels(method='GET', endpoint='/health', status=503).inc()
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    """Make single prediction"""
    request_start_time = time.time()
    
    try:
        # Track input feature stats
        for feature in ['temperature', 'humidity', 'wind_speed']:
            FEATURE_STATS.labels(
                city=input_data.city,
                feature=feature,
                stat_type='value'
            ).set(getattr(input_data, feature))
        
        # Process input data
        with DATA_PROCESSING_LATENCY.labels(city=input_data.city).time():
            data = pd.DataFrame([input_data.dict()])
            check_data_quality(data, input_data.city)
            
            # Add time-based features
            current_time = datetime.now()
            data['hour'] = current_time.hour
            data['day_of_week'] = current_time.weekday()
            data['month'] = current_time.month
            data['is_weekend'] = int(current_time.weekday() >= 5)
            
            # Calculate actual AQI
            actual_aqi = calculate_aqi(input_data.pm2_5, input_data.pm10)
            
            # Fill missing values
            if input_data.temperature_rolling_mean_6h is None:
                data['temperature_rolling_mean_6h'] = data['temperature']
            if input_data.humidity_rolling_mean_6h is None:
                data['humidity_rolling_mean_6h'] = data['humidity']
            if input_data.aqi_lag_1h is None:
                data['aqi_lag_1h'] = actual_aqi
            if input_data.aqi_lag_3h is None:
                data['aqi_lag_3h'] = actual_aqi
            if input_data.aqi_lag_6h is None:
                data['aqi_lag_6h'] = actual_aqi
        
        # Make prediction
        with MODEL_PREDICTION_LATENCY.labels(city=input_data.city).time():
            predictions = predictor.predict(data)
        
        # Update metrics
        predicted_aqi = float(predictions['predicted_aqi'].iloc[0])
        PREDICTED_AQI.labels(city=input_data.city).set(predicted_aqi)
        
        # Calculate and update prediction error
        prediction_error = abs(predicted_aqi - actual_aqi)
        PREDICTION_ERROR.labels(city=input_data.city).set(prediction_error)
        
        # Update model metrics
        confidence_range = float(predictions['confidence_upper'].iloc[0]) - float(predictions['confidence_lower'].iloc[0])
        confidence_percentage = max(0, min(1, 1 - (confidence_range / (predicted_aqi + 1e-6))))
        MODEL_METRICS.labels(metric_name='confidence_range').set(confidence_percentage)
        MODEL_METRICS.labels(metric_name='prediction_error').set(prediction_error)



           
        
        response = {
            "predicted_aqi": predicted_aqi,
            "confidence_lower": float(predictions['confidence_lower'].iloc[0]),
            "confidence_upper": float(predictions['confidence_upper'].iloc[0]),
            "risk_level": str(predictions['risk_level'].iloc[0]),
            "timestamp": current_time.isoformat()
        }
        
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status=200).inc()
        REQUEST_LATENCY.labels(method='POST', endpoint='/predict').observe(time.time() - request_start_time)
        API_HEALTH.labels(endpoint="/predict").set(1)
        return response
        
    except Exception as e:
        API_HEALTH.labels(endpoint="/predict").set(0)
        ERROR_RATE.labels(city=input_data.city, error_type='prediction_error').inc()
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status=500).inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/future")
async def predict_future(input_data: PredictionInput, hours_ahead: int = 24):
    """Make predictions for future hours"""
    request_start_time = time.time()
    
    try:
        if hours_ahead > 72:
            REQUEST_COUNT.labels(method='POST', endpoint='/predict/future', status=400).inc()
            raise HTTPException(status_code=400, detail="Cannot predict more than 72 hours ahead")
        
        # Process input data
        with DATA_PROCESSING_LATENCY.labels(city=input_data.city).time():
            data = pd.DataFrame([input_data.dict()])
            check_data_quality(data, input_data.city)
            
            current_time = datetime.now()
            data['hour'] = current_time.hour
            data['day_of_week'] = current_time.weekday()
            data['month'] = current_time.month
            data['is_weekend'] = int(current_time.weekday() >= 5)
            
            actual_aqi = calculate_aqi(input_data.pm2_5, input_data.pm10)
            
            # Fill missing values
            if input_data.temperature_rolling_mean_6h is None:
                data['temperature_rolling_mean_6h'] = data['temperature']
            if input_data.humidity_rolling_mean_6h is None:
                data['humidity_rolling_mean_6h'] = data['humidity']
            if input_data.aqi_lag_1h is None:
                data['aqi_lag_1h'] = actual_aqi
            if input_data.aqi_lag_3h is None:
                data['aqi_lag_3h'] = actual_aqi
            if input_data.aqi_lag_6h is None:
                data['aqi_lag_6h'] = actual_aqi
        
        # Make future predictions
        with MODEL_PREDICTION_LATENCY.labels(city=input_data.city).time():
            future_predictions = predictor.predict_future(data, hours_ahead)
        
        response = []
        for idx, row in future_predictions.iterrows():
            pred_aqi = float(row['predicted_aqi'])
            PREDICTED_AQI.labels(city=f"{input_data.city}_h{int(row['hours_ahead'])}").set(pred_aqi)
            
            prediction = {
                "hours_ahead": int(row['hours_ahead']),
                "predicted_aqi": pred_aqi,
                "confidence_lower": float(row['confidence_lower']),
                "confidence_upper": float(row['confidence_upper']),
                "risk_level": str(row['risk_level']),
                "timestamp": (current_time + pd.Timedelta(hours=int(row['hours_ahead']))).isoformat()
            }
            response.append(prediction)
        
        REQUEST_COUNT.labels(method='POST', endpoint='/predict/future', status=200).inc()
        REQUEST_LATENCY.labels(method='POST', endpoint='/predict/future').observe(time.time() - request_start_time)
        API_HEALTH.labels(endpoint="/predict/future").set(1)
        return response
        
    except Exception as e:
        API_HEALTH.labels(endpoint="/predict/future").set(0)
        ERROR_RATE.labels(city=input_data.city, error_type='future_prediction_error').inc()
        REQUEST_COUNT.labels(method='POST', endpoint='/predict/future', status=500).inc()
        logger.error(f"Future prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cities")
async def get_cities():
    """Get list of supported cities"""
    start_time = time.time()
    try:
        API_HEALTH.labels(endpoint="/cities").set(1)
        response = {
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
        REQUEST_COUNT.labels(method='GET', endpoint='/cities', status=200).inc()
        REQUEST_LATENCY.labels(method='GET', endpoint='/cities').observe(time.time() - start_time)
        return response
    except Exception as e:
        API_HEALTH.labels(endpoint="/cities").set(0)
        REQUEST_COUNT.labels(method='GET', endpoint='/cities', status=500).inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def get_model_info():
    """Get information about the current model"""
    start_time = time.time()
    try:
        if predictor is None:
            API_HEALTH.labels(endpoint="/model/info").set(0)
            REQUEST_COUNT.labels(method='GET', endpoint='/model/info', status=503).inc()
            raise HTTPException(status_code=503, detail="Model not initialized")
            
        API_HEALTH.labels(endpoint="/model/info").set(1)
        
        # Update model metrics
        MODEL_METRICS.labels(metric_name="training_rmse").set(
            float(predictor.metadata["metrics"].get("rmse", 0))
        )
        
        response = {
            "model_type": predictor.metadata["model_name"],
            "last_trained": predictor.metadata["timestamp"],
            "metrics": predictor.metadata["metrics"],
            "features": predictor.feature_columns,
            "performance": {
                "average_prediction_time": MODEL_PREDICTION_LATENCY._metrics[0].get_sample_sum() / max(MODEL_PREDICTION_LATENCY._metrics[0].get_sample_count(), 1),
                "total_predictions": sum(counter.get() for counter in REQUEST_COUNT._metrics if counter._labelvalues[1] == '/predict'),
                "error_rate": sum(counter.get() for counter in ERROR_RATE._metrics)
            }
        }
        
        REQUEST_COUNT.labels(method='GET', endpoint='/model/info', status=200).inc()
        REQUEST_LATENCY.labels(method='GET', endpoint='/model/info').observe(time.time() - start_time)
        return response
        
    except Exception as e:
        REQUEST_COUNT.labels(method='GET', endpoint='/model/info', status=500).inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/summary")
async def get_metrics_summary():
    """Get summary of all monitoring metrics"""
    start_time = time.time()
    try:
        API_HEALTH.labels(endpoint="/metrics/summary").set(1)
        
        # Calculate summary statistics
        total_predictions = sum([
            counter.get()
            for counter in REQUEST_COUNT._metrics
            if counter._labelvalues[1] == '/predict'
        ])
        
        average_latency = sum([
            hist.get_sample_sum() / max(hist.get_sample_count(), 1)
            for hist in REQUEST_LATENCY._metrics
            if hist._labelvalues[1] == '/predict'
        ])
        
        total_errors = sum([
            counter.get()
            for counter in ERROR_RATE._metrics
        ])
        
        average_prediction_error = sum([
            gauge.get()
            for gauge in PREDICTION_ERROR._metrics
        ]) / max(len(PREDICTION_ERROR._metrics), 1)
        
        response = {
            "api_health": {
                endpoint._labelvalues[0]: "UP" if gauge.get() > 0 else "DOWN"
                for endpoint, gauge in zip(API_HEALTH._metrics, API_HEALTH._metrics)
            },
            "performance_metrics": {
                "total_predictions": total_predictions,
                "average_latency_seconds": average_latency,
                "total_errors": total_errors,
                "error_rate": total_errors / max(total_predictions, 1),
                "average_prediction_error": average_prediction_error
            },
            "data_quality": {
                city._labelvalues[0]: {
                    "missing_values": gauge.get()
                    for gauge in DATA_QUALITY._metrics
                    if gauge._labelvalues[0] == city._labelvalues[0]
                    and gauge._labelvalues[1] == "missing_values"
                }
                for city in set(gauge._labelvalues[0] for gauge in DATA_QUALITY._metrics)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        REQUEST_COUNT.labels(method='GET', endpoint='/metrics/summary', status=200).inc()
        REQUEST_LATENCY.labels(method='GET', endpoint='/metrics/summary').observe(time.time() - start_time)
        return response
        
    except Exception as e:
        API_HEALTH.labels(endpoint="/metrics/summary").set(0)
        REQUEST_COUNT.labels(method='GET', endpoint='/metrics/summary', status=500).inc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)