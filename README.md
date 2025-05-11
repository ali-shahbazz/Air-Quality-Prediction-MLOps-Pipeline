Air Quality Monitoring & Prediction

This project implements an end-to-end MLOps pipeline for air quality monitoring and AQI (Air Quality Index) prediction using real-world environmental data. It covers data collection, processing, model training, deployment, monitoring, and automation.

## Project Structure
- **data/**: Raw and processed datasets, DVC versioning
- **src/**: Source code for API, data processing, and model training
- **models/**: Trained model artifacts and metadata
- **scripts/**: Automation scripts (e.g., scheduled data collection)
- **monitoring/**: Prometheus, Grafana, and alerting configs
- **docker/**: Dockerfile and docker-compose for containerized deployment
- **mlruns/**, **mlartifacts/**: MLflow experiment tracking
- **logs/**: Log files

## Main Features
- **Automated Data Collection**: Scheduled with DVC and Git integration
- **Data Processing**: Cleaning, feature engineering, normalization
- **Model Training**: Random Forest, Gradient Boosting, and Feedforward NN (PyTorch)
- **API**: FastAPI-based prediction service with Prometheus metrics
- **Monitoring**: Prometheus, Grafana dashboards, and alerting
- **Experiment Tracking**: MLflow for model metrics and artifacts
- **Containerization**: Docker support for reproducible deployment

## Setup Instructions
1. **Clone the repository**
2. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Configure DVC remote** (if using DVC for data versioning)
   ```powershell
   dvc remote add -d storage <remote-url>
   dvc pull
   ```
4. **Run data collection**
   ```powershell
   python scripts/schedule_collection.py
   ```
5. **Process data**
   ```powershell
   python src/data/preprocessor.py
   ```
6. **Train models**
   ```powershell
   python src/models/train.py
   # or for neural network
   python src/models/train2.py
   ```
7. **Start the API**
   ```powershell
   uvicorn src.api.app:app --reload
   ```
8. **(Optional) Run with Docker**
   ```powershell
   docker build -t mlops-aqi .
   docker run -p 8000:8000 mlops-aqi
   ```

## Monitoring & Visualization
- **Prometheus**: Collects API and model metrics
- **Grafana**: Visualizes metrics (see `monitoring/grafana/dashboards/`)
- **Alertmanager**: Sends alerts based on rules in `monitoring/prometheus/alert_rules.yml`

## API Endpoints
- `/predict` : Get AQI predictions
- `/model/info` : Model metadata and metrics
- `/metrics` : Prometheus metrics endpoint

## Documentation
- See `i210736_MLOPS_Project_Documentation.pdf` for detailed design and implementation notes.
- For project requirements, see `mlops_project_description.pdf` (if available).

---
*Developed for Fall 2024 MLOps Course.*
