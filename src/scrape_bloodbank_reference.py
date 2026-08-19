"""Collect public blood-bank directory metadata.

This script scrapes public directory information from a specified URL table
or uses historical reference data to establish a catalog of blood banks.
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


class TableParser(HTMLParser):
    """Simple parser to extract rows from HTML tables."""
    
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.current_row: list[str] = []
        self.current_cell: list[str] = []
        self.in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.current_row = []
        elif tag in {"td", "th"}:
            self.in_cell = True
            self.current_cell = []

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.current_cell.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"}:
            self.current_row.append(" ".join(x for x in self.current_cell if x))
            self.in_cell = False
        elif tag == "tr":
            if self.current_row:
                self.rows.append(self.current_row)


def get_fallback_rows() -> list[list[str]]:
    """Returns fallback mock data for testing offline or when no URL is provided."""
    return [
        ["bank_001", "Synthetic Reference Centre North", "Kanpur", "Uttar Pradesh", "26.4499", "80.3319"],
        ["bank_002", "Synthetic Reference Centre South", "Kanpur", "Uttar Pradesh", "26.4310", "80.2900"],
        ["bank_003", "Synthetic Reference Centre East", "Lucknow", "Uttar Pradesh", "26.8467", "80.9462"],
        ["bank_004", "Synthetic Reference Centre West", "Lucknow", "Uttar Pradesh", "26.8310", "80.9000"],
        ["bank_005", "Synthetic Reference Centre Central", "Prayagraj", "Uttar Pradesh", "25.4358", "81.8463"],
        ["bank_006", "Synthetic Reference Centre River", "Varanasi", "Uttar Pradesh", "25.3176", "82.9739"],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect public blood-bank directory metadata.")
    parser.add_argument(
        "--url", 
        help="Public URL containing a table with blood bank metadata (id, name, city, state, latitude, longitude)"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=DATA_DIR / "bloodbank_reference_catalog.csv",
        help="Output CSV file path"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = get_fallback_rows()
    source = "OFFLINE_FICTIONAL_REFERENCE"

    if args.url:
        logging.info(f"Attempting to fetch blood-bank directory from URL: {args.url}")
        try:
            req = Request(
                args.url, 
                headers={"User-Agent": "BloodBridgeAI academic project contact@example.invalid"}
            )
            with urlopen(req, timeout=20) as response:
                html_content = response.read().decode("utf-8", errors="replace")
            
            html_parser = TableParser()
            html_parser.feed(html_content)
            
            candidates = [r for r in html_parser.rows if len(r) >= 6]
            if not candidates:
                raise ValueError("No table with at least six cells found on the supplied URL.")
            
            rows = candidates[1:]  # Skip header row
            source = args.url
            logging.info(f"Successfully scraped {len(rows)} rows from {args.url}")
        except Exception as e:
            logging.error(f"Failed to fetch data from URL: {e}. Falling back to default records.")
            rows = get_fallback_rows()
            source = "OFFLINE_FICTIONAL_REFERENCE_FALLBACK"

    # Write to CSV
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bank_id", 
            "bank_name", 
            "city", 
            "state", 
            "latitude", 
            "longitude", 
            "data_source", 
            "collected_at_utc"
        ])
        
        utc_now = datetime.now(timezone.utc).isoformat()
        for row in rows:
            writer.writerow(row[:6] + [source, utc_now])

    logging.info(f"Successfully wrote {len(rows)} reference rows to {args.output}")


if __name__ == "__main__":
    main()
