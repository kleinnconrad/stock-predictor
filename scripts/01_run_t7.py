import os
import sys

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yaml
from src.ingestion.xetra_t7 import fetch_xetra_t7
from src.processing.qualifier import filter_qualified_tickers

def main():
    os.makedirs('data/state', exist_ok=True)
    print("Initiating Xetra T7 Download...")
    
    with open('config/settings.yaml', 'r') as f:
        settings = yaml.safe_load(f)
        
    raw_df = fetch_xetra_t7(settings['xetra_t7_url'])
    if raw_df.empty:
        raise ValueError("Failed to fetch T7 tickers. Network or parsing error.")
    
    qualified_tickers = filter_qualified_tickers(raw_df)
    
    output_path = 'data/state/qualified_tickers.csv'
    pd.DataFrame(qualified_tickers).to_csv(output_path, index=False)
    print(f"Successfully saved {len(qualified_tickers)} qualified tickers to {output_path}")

if __name__ == "__main__":
    main()
