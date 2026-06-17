import os
import pandas as pd
from src.ingestion.xetra_t7 import fetch_xetra_t7_tickers
from src.processing.qualifier import filter_qualified_retail_stocks

def main():
    os.makedirs('data/state', exist_ok=True)
    print("Initiating Xetra T7 Download...")
    raw_df = fetch_xetra_t7_tickers()
    if raw_df.empty:
        raise ValueError("Failed to fetch T7 tickers. Network or parsing error.")
    
    qualified_tickers = filter_qualified_retail_stocks(raw_df)
    
    output_path = 'data/state/qualified_tickers.csv'
    pd.DataFrame({'Ticker': qualified_tickers}).to_csv(output_path, index=False)
    print(f"Successfully saved {len(qualified_tickers)} qualified tickers to {output_path}")

if __name__ == "__main__":
    main()
