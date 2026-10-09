import os
import sys

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
from config.settings import load_settings
from src.ingestion.xetra_t7 import fetch_xetra_t7
from src.processing.qualifier import filter_qualified_tickers
from src.orchestration.logging_setup import configure_logging

logger = logging.getLogger(__name__)

def main():
    configure_logging()
    os.makedirs('data/state', exist_ok=True)
    logger.info("Initiating Xetra T7 Download...")
    
    settings = load_settings()
    raw_df = fetch_xetra_t7(settings['xetra_t7_url'])
    if raw_df.empty:
        raise ValueError("Failed to fetch T7 tickers. Network or parsing error.")
    
    qualified_tickers = filter_qualified_tickers(raw_df)
    
    output_path = 'data/state/qualified_tickers.csv'
    pd.DataFrame(qualified_tickers).to_csv(output_path, index=False)
    logger.info(f"Successfully saved {len(qualified_tickers)} qualified tickers to {output_path}")

if __name__ == "__main__":
    main()
