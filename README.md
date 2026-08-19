# BloodBridge AI — Machine Learning Pipeline

This repository contains the machine learning pipeline for **BloodBridge AI**, designed to forecast **next-day blood inventory shortage risk** across regional blood banks. The project supports time-series forecasting, data aggregation, and custom predictive models to help optimize blood inventory levels.

---

## Project Overview

Maintaining optimal blood inventory levels is a critical challenge. Too much inventory leads to wastage due to expiry, while too little inventory creates life-threatening shortages. 

This project implements a complete machine learning pipeline:
1. **Catalog Collection**: Fetches public reference metadata (coordinates, states, cities) for historical blood banks.
2. **Simulation Modeling**: Simulates daily inventory transactions (donations, requests, expirations) incorporating seasonality, calendar effects, and emergency spikes.
3. **Data Cleaning & Engineering**: Groups and cleans raw transactions, constructs 7-day rolling indicators (averages, gaps, demand trends), and defines next-day shortage labels.
4. **Model Training & Comparison**: Fits and compares linear baselines (Logistic Regression) with non-linear tree ensembles (Random Forest Classifier).
5. **Inference Pipeline**: Exposes trained models for next-day batch inference.

---

## Directory Structure

```text
BloodBridge_AI/
├── data/
│   ├── raw/                 # Public reference catalog metadata
│   ├── interim/             # Generated synthetic inventory transactions
│   └── processed/           # Preprocessed final ML feature matrix
├── models/
│   └── blood_shortage_model.joblib  # Trained scikit-learn classifier pipeline
├── notebooks/               # Step-by-step Jupyter Research Notebooks
│   ├── 00_web_scraping_real_bloodbank_directory.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_final_dataset_export.ipynb
│   └── 05_ml_model_training_evaluation.ipynb
├── reports/
│   └── model_metrics.json   # Logged classification evaluation metrics
├── src/                     # Core python pipeline modules
│   ├── generate_synthetic_dataset.py
│   ├── predict.py
│   ├── run_notebooks.py
│   ├── scrape_bloodbank_reference.py
│   └── train.py
└── requirements.txt         # Project dependencies
```

---

## Getting Started

### 1. Environment Setup

It is recommended to run the project in a Python virtual environment:

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Install project dependencies
python -m pip install -r requirements.txt
```

### 2. Running the Pipeline

You can run the entire data generation, notebook evaluation, and model training pipeline using the following steps:

```powershell
# Step 1: Collect blood bank catalog metadata
python src/scrape_bloodbank_reference.py

# Step 2: Simulate daily transaction data
python src/generate_synthetic_dataset.py --seed 42

# Step 3: Run and update all Jupyter notebooks in-place
python src/run_notebooks.py

# Step 4: Train final Random Forest model
python src/train.py
```

### 3. Evaluating Predictions (Batch Inference)

To execute the batch inference pipeline on sample data:

```powershell
python src/predict.py --input data/test_prediction_input.csv
```

The script will output predictions in JSON format:

```json
[
    {
        "inventory_ratio": 0.8881,
        "day_of_week": 3,
        "month": 1,
        "emergency_event_flag": 0,
        "requests_received_rolling_7d": 10.33,
        "donations_received_rolling_7d": 22.67,
        "inventory_ratio_rolling_7d": 0.8391,
        "request_donation_gap_7d": -12.33,
        "city": "Kanpur",
        "blood_group": "A+",
        "shortage_risk_probability": 0.0,
        "shortage_prediction": 0
    }
]
```

### 4. Running the FastAPI REST Server

Launch the web API service locally using Uvicorn:

```powershell
uvicorn src.api:app --reload --port 8000
```

Access API endpoints and documentation:
- **Interactive Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`
- **Health Check**: `GET http://127.0.0.1:8000/health`
- **Metrics**: `GET http://127.0.0.1:8000/metrics`
- **Shortage Prediction**: `POST http://127.0.0.1:8000/predict`
- **Catalog**: `GET http://127.0.0.1:8000/catalog`

---

## Data Source Note

Public blood-bank location catalog information was collected from public reference sources as contextual metadata. Because actual clinical records, donor details, and live hospital stock levels are private and confidential, a reproducible synthetic dataset was simulated for academic ML validation and software development demonstration.

---

## Limitations

This repository is designed for academic demonstration and software prototyping purposes only. It should not be used in live clinical operations or real-world medical distribution decisions.
