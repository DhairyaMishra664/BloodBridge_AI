"""Generate a reproducible synthetic dataset representing blood bank operations.

This script simulates daily inventory transactions (donations, demand/requests,
expirations) across multiple blood bank locations based on typical statistics.
The output is used for training and testing ML models.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"

# Configuration constants
BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
GROUP_DISTRIBUTION = {
    "A+": 0.29, "A-": 0.06, 
    "B+": 0.30, "B-": 0.06, 
    "AB+": 0.08, "AB-": 0.02, 
    "O+": 0.16, "O-": 0.03
}


def get_fallback_catalog() -> pd.DataFrame:
    """Returns a fallback list of blood banks in case raw scraped catalog isn't found."""
    cities = ["Kanpur", "Kanpur", "Lucknow", "Lucknow", "Prayagraj", "Varanasi"]
    records = []
    for i, city in enumerate(cities, 1):
        records.append([
            f"bank_{i:03d}",
            f"Synthetic Centre {i}",
            city,
            "Uttar Pradesh",
            26.0 + (i / 100.0),
            80.0 + (i / 100.0)
        ])
    return pd.DataFrame(
        records,
        columns=["bank_id", "bank_name", "city", "state", "latitude", "longitude"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic operational blood bank dataset.")
    parser.add_argument(
        "--catalog", 
        type=Path, 
        default=RAW_DATA_DIR / "bloodbank_reference_catalog.csv",
        help="Input reference catalog CSV"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=INTERIM_DATA_DIR / "synthetic_daily_blood_inventory.csv",
        help="Output synthetic dataset CSV"
    )
    parser.add_argument("--start-date", default="2024-01-01", help="Simulation start date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=540, help="Number of days to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load directory catalog
    if args.catalog.exists():
        logging.info(f"Loading reference catalog from {args.catalog}")
        catalog = pd.read_csv(args.catalog)
    else:
        logging.warning(f"Catalog {args.catalog} not found. Using fallback mock catalog.")
        catalog = get_fallback_catalog()

    required_cols = ["bank_id", "bank_name", "city", "state", "latitude", "longitude"]
    catalog = catalog[required_cols].dropna().drop_duplicates("bank_id")
    
    # Initialize simulation environment
    rng = np.random.default_rng(args.seed)
    simulation_rows = []
    dates = pd.date_range(args.start_date, periods=args.days, freq="D")

    logging.info(f"Starting simulation for {len(catalog)} banks over {args.days} days...")

    for bank_idx, bank in catalog.reset_index(drop=True).iterrows():
        # Establish stable coefficients for each bank to reflect operational variance
        donor_factor = rng.uniform(0.85, 1.15)
        demand_factor = rng.uniform(0.85, 1.20)

        for group in BLOOD_GROUPS:
            group_weight = GROUP_DISTRIBUTION[group]
            # Storage capacity scales with blood group prevalence and bank scale
            capacity = int(80 + 220 * group_weight + 13 * bank_idx)
            # Initial inventory starts at a random occupancy level
            inventory = int(capacity * rng.uniform(0.34, 0.72))

            for date in dates:
                is_weekend = int(date.dayofweek >= 5)
                # Introduce annual seasonality wave (e.g. higher supply in winter, lower in holiday seasons)
                seasonality = 1.0 + 0.16 * np.sin(2 * np.pi * date.dayofyear / 365.25)
                
                # Check for emergency high-demand event
                is_emergency = int(rng.random() < 0.045)

                # Base expected values for donations and requests
                base_donations = 9.0 + 31.0 * group_weight
                base_requests = 6.5 + 27.0 * group_weight

                # Introduce stochastic daily perturbations to simulate real-world operational variance
                donor_noise = float(rng.lognormal(mean=0.0, sigma=0.22))
                demand_noise = float(rng.lognormal(mean=0.0, sigma=0.25))

                # Adjust for weekend, seasonality, and unobserved daily operational noise
                expected_donations = base_donations * donor_factor * (0.83 if is_weekend else 1.0) * seasonality * donor_noise
                expected_requests = base_requests * demand_factor * (1.15 if is_weekend else 1.0) * seasonality * demand_noise
                
                # Emergencies add variable surge demand
                if is_emergency:
                    expected_requests += float(rng.uniform(5.0, 16.0))

                # Sample discrete event counts using Poisson distribution
                donations = max(0, int(rng.poisson(max(0.1, expected_donations))))
                requests = max(0, int(rng.poisson(max(0.1, expected_requests))))
                
                # Expired units reflect stock aging plus occasional batch spoilage/audit losses
                expected_expirations = 0.5 + (inventory / 140.0)
                spoilage_loss = int(rng.poisson(3.0)) if (rng.random() < 0.025) else 0
                units_expired = max(0, int(rng.poisson(expected_expirations)) + spoilage_loss)

                # Update daily ending inventory level
                inventory = max(0, min(capacity, inventory + donations - requests - units_expired))

                simulation_rows.append({
                    "record_date": date.date().isoformat(),
                    "bank_id": bank.bank_id,
                    "bank_name": bank.bank_name,
                    "city": bank.city,
                    "state": bank.state,
                    "latitude": bank.latitude,
                    "longitude": bank.longitude,
                    "blood_group": group,
                    "inventory_units": inventory,
                    "storage_capacity_units": capacity,
                    "donations_received": donations,
                    "requests_received": requests,
                    "units_expired": units_expired,
                    "emergency_event_flag": is_emergency,
                    "data_origin": "SYNTHETIC_DATA_SET",
                    "generation_seed": args.seed
                })

    df_out = pd.DataFrame(simulation_rows)
    df_out.to_csv(args.output, index=False)
    logging.info(f"Successfully generated and wrote {len(df_out):,} records to {args.output}")


if __name__ == "__main__":
    main()
