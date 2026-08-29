# scripts/download_kaggle_data.py
import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    # 1. Dynamically find the directory where THIS script lives
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    # 2. Define your destination (e.g., a 'data' folder next to the script)
    # Alternatively, use: SCRIPT_DIR.parent / "data" to target your project root data folder
    OUTPUT_DIR = SCRIPT_DIR.parent / "data"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 3. Authenticate with Kaggle (Requires ~/.kaggle/kaggle.json)
    api = KaggleApi()
    api.authenticate()
    
    # 4. Download the dataset
    # Replace with your target dataset (e.g., 'clmentbisaillon/fake-and-real-news-dataset')
    dataset = "varsharam/walmart-sales-dataset-of-45stores"
    print(f"Downloading {dataset} to {OUTPUT_DIR}...")
    
    api.dataset_download_files(dataset, path=str(OUTPUT_DIR), unzip=True)
    print("Download complete!")

if __name__ == "__main__":
    main()
