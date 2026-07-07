import os
import sys
import time
import random
import logging
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import requests

# Add the src and root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.universe import FUNDAMENTAL_UNIVERSE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_all_fundamentals():
    state_file = os.path.join('data', 'state', 'qualified_tickers.csv')
    if not os.path.exists(state_file):
        logger.error(f"State file {state_file} not found. Please run Step 1 first to generate qualified tickers.")
        return
        
    df = pd.read_csv(state_file)
    if 'Ticker' not in df.columns:
        logger.error("No 'Ticker' column found in qualified_tickers.csv.")
        return
        
    tickers = df['Ticker'].unique().tolist()
    
    out_dir = os.path.join('data', 'raw', 'fundamentals')
    os.makedirs(out_dir, exist_ok=True)
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    })
    
    success_count = 0
    for ticker in tqdm(tickers, desc="Fetching Fundamentals"):
        tickers_to_try = [ticker]
        if ticker.endswith('.DE'):
            tickers_to_try.append(ticker[:-3] + '.F')
            
        success = False
        for t in tickers_to_try:
            if success:
                break
                
            max_retries = 3
            backoff_delay = 2.0
            
            for attempt in range(max_retries):
                jitter = random.uniform(2.0, 4.0)
                time.sleep(jitter)
                
                try:
                    stock = yf.Ticker(t, session=session)
                    
                    financials = stock.quarterly_financials.T
                    balance_sheet = stock.quarterly_balance_sheet.T
                    cashflow = stock.quarterly_cashflow.T
                    
                    if financials.empty and balance_sheet.empty and cashflow.empty:
                        logger.debug(f"Empty data returned for {t} on attempt {attempt+1}")
                        time.sleep(backoff_delay)
                        backoff_delay *= 2
                        continue
                        
                    fundamentals = pd.concat([financials, balance_sheet, cashflow], axis=1)
                    fundamentals = fundamentals.loc[:, ~fundamentals.columns.duplicated()]
                    
                    cols_to_keep = [c for c in FUNDAMENTAL_UNIVERSE if c in fundamentals.columns]
                    fundamentals = fundamentals[cols_to_keep]
                    
                    if fundamentals.empty:
                        logger.debug(f"Filtered data empty for {t} on attempt {attempt+1}")
                        time.sleep(backoff_delay)
                        backoff_delay *= 2
                        continue
                        
                    fundamentals.index = pd.to_datetime(fundamentals.index)
                    fundamentals = fundamentals.sort_index()
                    
                    # Save local cache using ticker with underscores to avoid Windows reserved name errors (e.g., CON.DE)
                    safe_ticker = ticker.replace('.', '_')
                    out_path = os.path.join(out_dir, f"{safe_ticker}.csv")
                    fundamentals.to_csv(out_path)
                    
                    logger.info(f"Successfully cached fundamentals for {ticker} (fetched via {t})")
                    success = True
                    success_count += 1
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch {t} on attempt {attempt+1}: {e}")
                    time.sleep(backoff_delay)
                    backoff_delay *= 2
                    
        if not success:
            logger.warning(f"Could not fetch fundamental data for {ticker} after all retries and fallback options.")
            
    logger.info(f"Update complete. Successfully fetched {success_count}/{len(tickers)} tickers.")

if __name__ == '__main__':
    update_all_fundamentals()
