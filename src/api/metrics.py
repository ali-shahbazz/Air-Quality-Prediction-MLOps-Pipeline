# src/api/metrics.py

from prometheus_client import Counter, Histogram, Gauge
import time

# Request count and latency metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP request count',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint']
)

# Prediction metrics
PREDICTION_ERROR = Gauge(
    'prediction_error',
    'Absolute difference between predicted and actual AQI'
)

PREDICTED_AQI = Gauge(
    'predicted_aqi',
    'Predicted AQI value',
    ['city']
)

DATA_COLLECTION_FAILURES = Counter(
    'data_collection_failures_total',
    'Total number of data collection failures'
)

MODEL_PREDICTION_COUNT = Counter(
    'model_predictions_total',
    'Total number of predictions made by the model'
)

class MetricsMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        start_time = time.time()
        
        # Modified send to capture the status code
        original_send = send
        status_code = None
        
        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await original_send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            if status_code:
                method = scope["method"]
                endpoint = scope["path"]
                
                # Record metrics
                REQUEST_COUNT.labels(
                    method=method,
                    endpoint=endpoint,
                    status=status_code
                ).inc()
                
                REQUEST_LATENCY.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(time.time() - start_time)

def track_prediction(city: str, predicted_value: float):
    """Track prediction metrics"""
    PREDICTED_AQI.labels(city=city).set(predicted_value)
    MODEL_PREDICTION_COUNT.inc()

def track_prediction_error(error: float):
    """Track prediction error"""
    PREDICTION_ERROR.set(error)

def track_data_collection_failure():
    """Track data collection failure"""
    DATA_COLLECTION_FAILURES.inc()