"""Load the trained model and return shortage-risk predictions on an input dataset."""

import argparse
import logging
from pathlib import Path
import joblib
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate inputs with trained shortage-risk model.")
    parser.add_argument(
        "--input", 
        required=True, 
        help="Path to CSV file containing the features for evaluation"
    )
    parser.add_argument(
        "--model", 
        type=Path, 
        default=ROOT / "models/blood_shortage_model.joblib",
        help="Path to the trained joblib model file"
    )
    args = parser.parse_args()

    # Load artifacts and evaluation dataset
    logging.info(f"Loading trained model artifact from: {args.model}")
    model_artifact = joblib.load(args.model)
    
    input_data = pd.read_csv(args.input)
    required_features = model_artifact["features"]
    
    # Check for missing feature columns
    missing_cols = [col for col in required_features if col not in input_data.columns]
    if missing_cols:
        raise ValueError(f"Input data is missing the following required columns: {missing_cols}")

    # Predict probabilities and final predictions
    logging.info("Running inference pipeline...")
    pipeline = model_artifact["pipeline"]
    threshold = model_artifact["threshold"]

    probabilities = pipeline.predict_proba(input_data[required_features])[:, 1]

    # Structure outputs
    results = input_data.copy()
    results["shortage_risk_probability"] = probabilities.round(4)
    results["shortage_prediction"] = (probabilities >= threshold).astype(int)

    # Print results as structured JSON format
    print(results.to_json(orient="records", indent=4))


if __name__ == "__main__":
    main()
