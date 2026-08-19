"""Execute the Jupyter notebooks in-place to populate their output cells and plots."""

import logging
from pathlib import Path
import nbformat
from nbclient import NotebookClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"


def main() -> None:
    # Find all notebook files
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    
    if not notebooks:
        raise FileNotFoundError(f"No notebooks found in directory: {NOTEBOOKS_DIR}")

    for nb_path in notebooks:
        logging.info(f"Executing notebook: {nb_path.name}...")
        try:
            # Read notebook content
            nb = nbformat.read(nb_path, as_version=4)
            
            # Configure execution client
            client = NotebookClient(
                nb, 
                timeout=600, 
                kernel_name="python3", 
                resources={"metadata": {"path": str(ROOT)}}
            )
            
            # Execute in-place
            client.execute()
            
            # Write back executed notebook
            nbformat.write(nb, nb_path)
            logging.info(f"Successfully executed and saved: {nb_path.name}")
        except Exception as e:
            logging.error(f"Error executing notebook {nb_path.name}: {e}")
            logging.warning("Proceeding to the next notebook...")

    logging.info("Notebook execution workflow completed.")


if __name__ == "__main__":
    main()
