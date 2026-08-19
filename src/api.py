"""FastAPI REST API service for BloodBridge AI inventory shortage risk forecasting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "blood_shortage_model.joblib"
METRICS_PATH = ROOT / "reports" / "model_metrics.json"
CATALOG_PATH = ROOT / "data" / "bloodbank_reference_catalog.csv"

app = FastAPI(
    title="BloodBridge AI — Inventory Shortage Forecasting API",
    description="REST API for next-day blood shortage risk prediction across regional blood banks.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
_MODEL_DATA: dict[str, Any] | None = None


def get_model():
    """Loads and caches the scikit-learn model pipeline."""
    global _MODEL_DATA
    if _MODEL_DATA is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model artifact not found at {MODEL_PATH}. Train the model first.")
        logging.info(f"Loading trained model artifact from {MODEL_PATH}")
        _MODEL_DATA = joblib.load(MODEL_PATH)
    return _MODEL_DATA


class PredictionInput(BaseModel):
    inventory_ratio: float = Field(..., ge=0.0, le=1.0, description="Current day ending inventory ratio (0.0 - 1.0)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(..., ge=1, le=12, description="Month of year (1-12)")
    emergency_event_flag: int = Field(0, ge=0, le=1, description="1 if local emergency event active, else 0")
    requests_received_rolling_7d: float = Field(..., ge=0.0, description="7-day rolling average of requests received")
    donations_received_rolling_7d: float = Field(..., ge=0.0, description="7-day rolling average of donations received")
    inventory_ratio_rolling_7d: float = Field(..., ge=0.0, le=1.0, description="7-day rolling average inventory ratio")
    request_donation_gap_7d: float = Field(..., description="7-day rolling gap (requests - donations)")
    city: str = Field("Kanpur", description="City location of blood bank")
    blood_group: str = Field("A+", description="ABO/Rh blood group (e.g., A+, O-, B+)")


class PredictionOutput(PredictionInput):
    shortage_risk_probability: float = Field(..., description="Estimated next-day shortage probability [0.0, 1.0]")
    shortage_prediction: int = Field(..., description="Binary prediction: 1 = Shortage Risk, 0 = Adequate Stock")


class BatchPredictionRequest(BaseModel):
    items: List[PredictionInput]


@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint returns operational status and model state."""
    model_loaded = MODEL_PATH.exists()
    return {
        "status": "online",
        "service": "BloodBridge AI REST API",
        "version": "1.0.0",
        "model_artifact_present": model_loaded
    }


@app.get("/metrics", tags=["Metrics"])
def get_model_metrics():
    """Returns the latest evaluated classification metrics from training report."""
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics report not found. Train the model first.")
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics


@app.get("/catalog", tags=["Reference Catalog"])
def get_catalog():
    """Returns blood bank reference catalog metadata."""
    if not CATALOG_PATH.exists():
        raise HTTPException(status_code=404, detail="Catalog file not found.")
    df_cat = pd.read_csv(CATALOG_PATH)
    return df_cat.to_dict(orient="records")


@app.post("/predict", response_model=List[PredictionOutput], tags=["Inference"])
def predict_shortage(payload: Union[PredictionInput, List[PredictionInput], BatchPredictionRequest]):
    """Predict next-day blood shortage risk for single or batch inputs."""
    model_data = get_model()
    pipeline = model_data["pipeline"]
    feature_cols = model_data["features"]
    threshold = model_data.get("threshold", 0.5)

    # Extract items list
    if isinstance(payload, BatchPredictionRequest):
        items = payload.items
    elif isinstance(payload, list):
        items = payload
    else:
        items = [payload]

    if not items:
        raise HTTPException(status_code=400, detail="Input items list cannot be empty.")

    input_dicts = [item.model_dump() for item in items]
    df_in = pd.DataFrame(input_dicts)

    # Validate missing feature columns
    for col in feature_cols:
        if col not in df_in.columns:
            raise HTTPException(status_code=400, detail=f"Missing required feature column: {col}")

    X = df_in[feature_cols].copy()
    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = []
    for item, prob, pred in zip(input_dicts, probabilities, predictions):
        item["shortage_risk_probability"] = round(float(prob), 4)
        item["shortage_prediction"] = int(pred)
        results.append(item)

    return results
