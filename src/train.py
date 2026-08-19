"""Train the shortage-risk prediction model using a leakage-safe time split."""

import argparse
import json
import logging
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data/processed/final_feature_engineered_bloodbridge_dataset.csv"

# Define variables
FEATURE_COLUMNS = [
    "inventory_ratio",
    "day_of_week",
    "month",
    "emergency_event_flag",
    "requests_received_rolling_7d",
    "donations_received_rolling_7d",
    "inventory_ratio_rolling_7d",
    "request_donation_gap_7d",
    "city",
    "blood_group"
]
TARGET_COLUMN = "shortage_next_day"


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Loads and returns the preprocessed and sorted CSV dataset."""
    logging.info(f"Loading feature dataset from: {data_path}")
    df = pd.read_csv(data_path, parse_dates=["record_date"])
    # Sort chronologically to prevent future lookahead leakage during splits
    df = df.sort_values("record_date").reset_index(drop=True)
    return df


def build_preprocessing_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Creates the standard scikit-learn preprocessor pipeline."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, numeric_cols),
        ("categorical", categorical_transformer, categorical_cols)
    ])
    
    return preprocessor


def evaluate_model(y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> dict:
    """Calculates classification metrics on evaluation set."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    roc_auc = None
    if y_true.nunique() > 1:
        roc_auc = round(float(roc_auc_score(y_true, y_prob)), 4)

    metrics = {
        "model": "random_forest_classifier",
        "evaluation": "global_chronological_holdout_80_20",
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BloodBridge shortage forecasting model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Path to features CSV")
    parser.add_argument("--test-fraction", type=float, default=0.20, help="Holdout split test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    args = parser.parse_args()

    # Load dataset
    df = load_dataset(args.data)
    
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)

    # Perform chronological hold-out split
    split_index = int(len(df) * (1 - args.test_fraction))
    
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    logging.info(f"Training sample size: {len(X_train)} | Test sample size: {len(X_test)}")

    # Define numeric and categorical columns
    categorical_cols = ["city", "blood_group"]
    numeric_cols = [col for col in FEATURE_COLUMNS if col not in categorical_cols]

    # Preprocessor and Estimator definition
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1
    )

    # Build and fit pipeline
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    logging.info("Training classifier model pipeline...")
    pipeline.fit(X_train, y_train)

    # Predict test sets
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    # Compute evaluation metrics
    metrics = evaluate_model(y_test, predictions, probabilities)

    # Ensure folders exist
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)

    # Dump output artifacts
    model_path = ROOT / "models/blood_shortage_model.joblib"
    report_path = ROOT / "reports/model_metrics.json"

    joblib.dump(
        {"pipeline": pipeline, "features": FEATURE_COLUMNS, "threshold": 0.5}, 
        model_path
    )
    
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    logging.info(f"Successfully trained and saved model pipeline to: {model_path}")
    logging.info(f"Evaluation Metrics saved to: {report_path}")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()
