# Mobile Reviews Sentiment Analysis - MLSecOps Pipeline
A complete MLSecOps pipeline for predicting mobile phone review sentiment (Positive, Negative, Neutral) using XGBoost, with MLflow experiment tracking, Airflow orchestration, Docker containerization, and Evidently AI monitoring.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Monitoring](#monitoring)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements a machine learning pipeline that predicts sentiment (Positive, Negative, Neutral) for mobile phone reviews based on various features including ratings, price, brand, customer demographics, and review metadata. The system integrates:

- **Machine Learning**: XGBoost multi-class classifier
- **Experiment Tracking**: MLflow for model versioning and artifact management
- **Orchestration**: Apache Airflow for automated training workflows
- **Containerization**: Docker for consistent deployment environments
- **Monitoring**: Evidently AI for data quality and model performance reports
- **Visualization**: Streamlit dashboard for interactive report viewing
- **REST API**: Flask-RESTX for model serving and inference

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Airflow       │────▶│   Flask API     │────▶│   MLflow        │
│   (Scheduler)   │     │   (Training)    │     │   (Tracking)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │                         │
                               ▼                         ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   Evidently     │     │   Streamlit     │
                        │   (Reports)     │────▶│   (Dashboard)   │
                        └─────────────────┘     └─────────────────┘
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| Flask API | 8080 | Model training and inference endpoints |
| MLflow Server | 5102 | Experiment tracking and model registry |
| Airflow Webserver | 8090 | DAG orchestration UI |
| Streamlit | 8501 | Evidently report visualization |
| PostgreSQL | 5432 | Airflow metadata database |

## Features

- **Automated Training Pipeline**: Airflow DAG triggers model training and promotes models based on accuracy improvements
- **Model Versioning**: MLflow tracks experiments, parameters, metrics, and artifacts
- **Staging/Production Workflow**: Models are tagged as "Staging" and promoted to "Production" based on performance
- **Data Quality Monitoring**: Evidently generates comprehensive data summary and classification reports
- **Multi-class Classification**: Predicts three sentiment classes (Positive, Negative, Neutral)
- **REST API**: 
  - `/model/train` - Train new models with CSV data
  - `/model/predict` - Make predictions on new reviews
  - `/model/register` - Register trained models in MLflow
  - `/model/status` - Get current model status
- **Interactive Dashboard**: Streamlit app displays Evidently reports with data quality and model metrics
- **Containerized Deployment**: All services run in Docker containers with shared volumes

## Prerequisites

- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 1.29 or higher
- **Python**: 3.12 (for local development)
- **Storage**: At least 5GB free disk space

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd mobile_sentiment_mlops
```

### 2. Build Docker Image

```bash
docker build -t mobile-sentiment-mlsecops .
```

This builds the main image containing:
- Python 3.12
- All required packages (XGBoost, MLflow, Evidently, Flask, Streamlit)
- Application code (app.py, MLModel.py, constants.py, streamlit_app.py)
- MLflow server configured to run on port 5102

### 3. Start All Services

```bash
docker-compose -f docker-compose-airflow.yml up -d
```

This starts:
- PostgreSQL database
- Airflow webserver, scheduler, and init container
- MLflow server and Flask API
- Streamlit dashboard

### 4. Verify Services

Wait 30-60 seconds for services to initialize, then check:

```bash
docker-compose -f docker-compose-airflow.yml ps
```

All services should show "Up" status.

### 5. Access Web Interfaces

- **Airflow UI**: http://localhost:8090 (admin/admin)
- **MLflow UI**: http://localhost:5102
- **Streamlit Dashboard**: http://localhost:8501
- **API Documentation**: http://localhost:8080

## Usage

### Training a Model

#### Option 1: Via Airflow DAG (Recommended)

1. Navigate to http://localhost:8090
2. Login with `admin/admin`
3. Find the `train_sentiment_model_dag` DAG
4. Click the play button to trigger the DAG
5. The DAG will:
   - Train a new model with the dataset in `/opt/airflow/data/`
   - Compare accuracy with the current Staging model
   - Promote the new model to Staging if it's better
   - Promote to Production if accuracy > 0.80
   - Send notifications on completion

#### Option 2: Via REST API

```bash
curl -X POST http://localhost:8080/model/train \
  -F "file=@data/Mobile_Reviews_Sentiment.csv"
```

Response:
```json
{
  "message": "Model trained successfully!",
  "run_id": "abc123...",
  "train_accuracy": 0.8567,
  "test_accuracy": 0.8423,
  "classification_report": {
    "Positive": {"precision": 0.89, "recall": 0.87, "f1-score": 0.88},
    "Negative": {"precision": 0.82, "recall": 0.80, "f1-score": 0.81},
    "Neutral": {"precision": 0.84, "recall": 0.85, "f1-score": 0.84}
  }
}
```

### Making Predictions

```bash
curl -X POST http://localhost:8080/model/predict \
  -H "Content-Type: application/json" \
  -d '{
    "reviews": [
      {
        "customer_name": "John Doe",
        "age": 28,
        "brand": "Apple",
        "model": "iPhone 14",
        "price_usd": 999,
        "price_local": 999,
        "currency": "USD",
        "exchange_rate_to_usd": 1.0,
        "rating": 5,
        "country": "USA",
        "language": "English",
        "review_date": "2025-01-15",
        "verified_purchase": true,
        "battery_life_rating": 4,
        "camera_rating": 5,
        "performance_rating": 5,
        "design_rating": 5,
        "display_rating": 5,
        "review_length": 75,
        "word_count": 15,
        "helpful_votes": 10,
        "source": "Amazon"
      }
    ]
  }'
```

Response:
```json
{
  "predictions": ["Positive"],
  "message": "Predictions made successfully"
}
```

### Viewing Evidently Reports

1. Train a model (generates report automatically)
2. Navigate to http://localhost:8501
3. View the interactive data quality and model performance report
4. Report includes:
   - Dataset statistics
   - Missing values analysis
   - Feature distributions
   - Classification metrics
   - Confusion matrix
   - Per-class performance

### Checking Model Status

```bash
curl -X GET http://localhost:8080/model/status
```

Response:
```json
{
  "staging_model_loaded": true,
  "staging_run_id": "abc123...",
  "mlflow_tracking_uri": "file:///app/mlruns"
}
```

## API Documentation

### Interactive API Docs

Visit http://localhost:8080 for Swagger UI with interactive API documentation.

### Endpoints

#### `POST /model/train`

Train a new XGBoost sentiment classifier with provided CSV data.

**Request**: Multipart form data with CSV file
- `file`: CSV file with training data (must include all required columns)

**Response**:
```json
{
  "message": "Model trained successfully!",
  "run_id": "abc123",
  "train_accuracy": 0.86,
  "test_accuracy": 0.84,
  "classification_report": {...}
}
```

#### `POST /model/predict`

Make predictions on new reviews using the current Staging model.

**Request**: JSON
```json
{
  "reviews": [
    {
      "customer_name": "...",
      "age": 30,
      "brand": "Samsung",
      ...
    }
  ]
}
```

**Response**:
```json
{
  "predictions": ["Positive", "Negative", "Neutral"],
  "message": "Predictions made successfully"
}
```

#### `POST /model/register`

Register a trained model in MLflow model registry.

**Request**: JSON
```json
{
  "run_id": "run_id_from_mlflow",
  "model_name": "Mobile_Sentiment_Model"
}
```

#### `GET /model/status`

Get current model status and configuration.

**Response**:
```json
{
  "staging_model_loaded": true,
  "staging_run_id": "abc123",
  "mlflow_tracking_uri": "file:///app/mlruns"
}
```

## Project Structure

```
mobile_sentiment_mlops/
├── app.py                              # Flask REST API application
├── MLModel.py                          # ML model class with training/inference
├── constants.py                        # Feature definitions and hyperparameters
├── streamlit_app.py                    # Streamlit dashboard for reports
├── Dockerfile                          # Docker image definition
├── docker-compose-airflow.yml          # Multi-container orchestration
├── requirements.txt                    # Python dependencies (local dev)
├── requirements_docker.txt             # Python dependencies (Docker)
├── .gitignore                          # Git ignore file
├── dags/
│   └── train_sentiment_model_dag.py   # Airflow DAG for automated training
├── data/
│   └── Mobile_Reviews_Sentiment_cleaned.csv   # Training dataset (50,000 reviews)
├── tests/
│   └── test_train_inference.py        # Unit tests
├── mlruns/                            # MLflow experiment data
├── mlartifacts/                       # MLflow artifact storage
└── logs/                              # Airflow logs
```

## Model Details

### Dataset

The **Mobile_Reviews_Sentiment.csv** dataset contains 50,000 mobile phone reviews with the following features:

### Algorithm

**XGBoost Multi-class Classifier** with the following hyperparameters:
- `max_depth`: 6
- `n_estimators`: 100
- `learning_rate`: 0.1
- `objective`: 'multi:softmax'
- `num_class`: 3
- `random_state`: 42

### Features (23 features + derived temporal features)

#### Categorical Features (7)
- `brand`: Phone brand (Apple, Samsung, Google, Xiaomi, OnePlus, Motorola, Realme)
- `model`: Phone model
- `currency`: Purchase currency
- `country`: Customer country
- `language`: Review language
- `source`: Review source (Amazon, eBay, Flipkart, etc.)
- `customer_name`: Customer identifier

#### Numerical Features (13)
- `age`: Customer age
- `price_usd`: Price in USD
- `price_local`: Price in local currency
- `exchange_rate_to_usd`: Exchange rate
- `rating`: Overall rating (1-5)
- `battery_life_rating`: Battery rating (1-5)
- `camera_rating`: Camera rating (1-5)
- `performance_rating`: Performance rating (1-5)
- `design_rating`: Design rating (1-5)
- `display_rating`: Display rating (1-5)
- `review_length`: Length of review text
- `word_count`: Number of words in review
- `helpful_votes`: Number of helpful votes

#### Binary Features (1)
- `verified_purchase`: Whether purchase was verified (True/False)

#### Temporal Features (Derived from review_date)
- `review_year`: Year of review
- `review_month`: Month of review
- `review_day`: Day of month
- `review_dayofweek`: Day of week

### Target Variable

- `sentiment`: Multi-class classification
  - **Positive**: Favorable reviews
  - **Negative**: Unfavorable reviews
  - **Neutral**: Mixed or moderate reviews

### Preprocessing Pipeline

1. **Drop ID and Text Columns**: Remove `review_id` and `review_text`
2. **Date Feature Engineering**: Extract year, month, day, dayofweek from `review_date`
3. **Label Encoding**: Transform categorical features to integers
4. **Standard Scaling**: Normalize numerical and temporal features
5. **Type Conversion**: Convert all features to float64

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Per-class Metrics**:
  - Precision
  - Recall
  - F1-score
- **Confusion Matrix**: Detailed classification breakdown

## Monitoring

### Evidently AI Reports

Automatically generated during training with:
- **DataSummaryPreset**: Comprehensive dataset statistics
  - Number of rows and columns
  - Missing values analysis
  - Column types and distributions
  - Feature correlations
- **ClassificationPreset**: Model performance metrics
  - Accuracy, precision, recall, F1-score
  - Confusion matrix
  - Per-class performance
  - Prediction distribution

Reports are:
1. Saved as MLflow artifacts
2. Stored in shared Docker volume (`/app/reports/report.html`)
3. Displayed in Streamlit dashboard

### MLflow Tracking

Each training run logs:
- **Parameters**: 
  - Model hyperparameters
  - Preprocessing configuration
  - Dataset size and features
- **Metrics**: 
  - Train/test accuracy
  - Per-class precision, recall, F1-score
- **Artifacts**: 
  - Trained XGBoost model
  - Label encoders (pickle)
  - Scaler (pickle)
  - Sentiment encoder (pickle)
  - Evidently report (HTML)
  - Confusion matrix

### Model Promotion Workflow

```
New Model Training
       ↓
Compare with Staging Model
       ↓
    Better?
    ↙    ↘
  Yes     No
   ↓       ↓
Promote  Keep Old
  ↓
Tag as "Staging"
  ↓
Accuracy > 0.80?
  ↙    ↘
Yes     No
 ↓       ↓
Prod   Staging Only
```

## Development

### Local Development Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all tests
pytest tests/test_train_inference.py -v

# Run specific test
pytest tests/test_train_inference.py::test_model_training -v
```

### Running Flask API Locally

```bash
# Start MLflow server
mlflow server --host 127.0.0.1 --port 5102 \
  --backend-store-uri file:///path/to/mlruns \
  --default-artifact-root /path/to/mlartifacts

# In another terminal, start Flask app
python app.py
```

API available at http://127.0.0.1:8080

### Running Streamlit Dashboard Locally

```bash
streamlit run streamlit_app.py
```

Dashboard available at http://localhost:8501

### Rebuilding Docker Image

After code changes:

```bash
# Rebuild image
docker build -t mobile-sentiment-mlsecops .

# Restart services
docker-compose -f docker-compose-airflow.yml down
docker-compose -f docker-compose-airflow.yml up -d
```

## Troubleshooting

### Issue: Streamlit shows "No report found"

**Solution**: Train a model first to generate the report
```bash
curl -X POST http://localhost:8080/model/train \
  -F "file=@data/Mobile_Reviews_Sentiment.csv"
```

### Issue: Docker containers won't start

**Solution**: Check logs
```bash
docker-compose -f docker-compose-airflow.yml logs <service-name>
```

Common causes:
- Port already in use: Change port mapping in docker-compose-airflow.yml
- Insufficient disk space: Clean up old images with `docker system prune`
- Permission issues: Ensure user has Docker permissions

### Issue: Airflow DAG not appearing

**Solution**: 
1. Check DAG file in `dags/` directory
2. Verify no syntax errors: 
   ```bash
   docker-compose -f docker-compose-airflow.yml exec airflow-webserver airflow dags list
   ```
3. Check DAG logs:
   ```bash
   docker-compose -f docker-compose-airflow.yml logs airflow-scheduler
   ```
4. Restart scheduler: 
   ```bash
   docker-compose -f docker-compose-airflow.yml restart airflow-scheduler
   ```

### Issue: Model prediction fails with "No staging model loaded"

**Solution**: Train and tag a model first
```bash
# Check model status
curl -X GET http://localhost:8080/model/status

# Train a model
curl -X POST http://localhost:8080/model/train \
  -F "file=@data/Mobile_Reviews_Sentiment.csv"
```

### Issue: MLflow artifacts not accessible

**Solution**: 
- Check volume mounts in docker-compose-airflow.yml
- Verify `/app/mlruns` and `/app/mlartifacts` directories exist in container:
  ```bash
  docker exec sentiment_mlsecops ls -la /app
  ```
- Ensure `--serve-artifacts` flag is set in MLflow server command

### Issue: Evidently import errors

**Solution**: 
1. Verify `evidently==0.4.29` is in requirements_docker.txt
2. Rebuild Docker image: `docker build -t mobile-sentiment-mlsecops .`
3. Restart containers:
   ```bash
   docker-compose -f docker-compose-airflow.yml down
   docker-compose -f docker-compose-airflow.yml up -d
   ```

### Issue: Memory errors during training

**Solution**: 
- Increase Docker memory allocation (Settings → Resources)
- Reduce dataset size for testing
- Adjust XGBoost parameters (reduce `n_estimators` or `max_depth`)

## Configuration

### Environment Variables

Set in docker-compose-airflow.yml:

```yaml
environment:
  MLFLOW_TRACKING_URI: "file:///app/mlruns"
  AIRFLOW__CORE__EXECUTOR: LocalExecutor
  AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
```

### Volumes

- `shared-mlruns`: MLflow experiment tracking data
- `shared-mlartifacts`: MLflow artifact storage
- `shared-reports`: Evidently reports shared between mlsecops and streamlit
- `postgres-db-volume`: PostgreSQL database persistence
- `./dags:/opt/airflow/dags`: Airflow DAG files
- `./data:/opt/airflow/data`: Training data
- `./logs:/opt/airflow/logs`: Airflow execution logs

### Hyperparameter Tuning

To adjust model hyperparameters, edit `constants.py`:

```python
XGBOOST_PARAMS = {
    'max_depth': 6,              # Tree depth
    'n_estimators': 100,         # Number of trees
    'learning_rate': 0.1,        # Learning rate
    'objective': 'multi:softmax',
    'num_class': 3,
    'random_state': 42
}
```

Then rebuild and restart services.

## Performance Benchmarks

Expected performance on the full dataset (50,000 reviews):

- **Training Time**: ~1-5 minutes (depends on hardware)
- **Test Accuracy**: ~85-95%
- **Per-class F1-scores**: ~0.80-0.88
- **Prediction Latency**: <100ms for single review

**Last Updated**: December 2025
