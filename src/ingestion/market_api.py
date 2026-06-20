import pandas as pd
import yfinance as yf
import datetime
import logging

logger = logging.getLogger(__name__)

def fetch_step1_data(ticker: str, global_macro_df: pd.DataFrame, history_years: int = 10) -> pd.DataFrame:
    """
    Fetches target stock OHLCV data and merges it with the global macro cache.
    
    Args:
        ticker (str): The stock ticker (e.g., 'SAP.DE').
        global_macro_df (pd.DataFrame): Pre-cached global macro dataset.
        history_years (int): Number of years of history to fetch.
        
    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    logger.info(f"Fetching market pricing for target {ticker}.")
    try:
        import time
        import random
        
        # Use Europe/Berlin time to ensure consistency between local and cloud runs
        current_german_date = pd.Timestamp.now(tz='Europe/Berlin').date()
        end_date = current_german_date
        start_date = end_date - datetime.timedelta(days=history_years * 365)
        
        df = pd.DataFrame()
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not df.empty:
                    # Explicitly strip intraday/live data to stabilize the model
                    df = df[df.index.date < current_german_date]
                    if not df.empty:
                        break
            except Exception as e:
                logger.warning(f"Exception during yf.download for {ticker} on attempt {attempt + 1}: {e}")
                
            logger.warning(f"Attempt {attempt + 1}/{max_retries}: No pricing data fetched for {ticker}. Retrying...")
            if attempt < max_retries - 1:
                # Exponential backoff with jitter to prevent concurrent thundering herd
                sleep_time = (2 ** attempt) + random.uniform(0.5, 2.0)
                time.sleep(sleep_time)
                
        if df.empty:
            logger.error(f"Failed to fetch pricing data for {ticker} after {max_retries} attempts.")
            return df
        
        # Flatten multi-index columns if present (yf >= 0.2.31 might return multi-index)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel('Ticker')
            
        df = df.ffill()
        
        # Ensure index matches
        df.index = pd.to_datetime(df.index)
        
        # Merge target stock OHLCV with the massive global macro dataframe
        merged_df = df.join(global_macro_df, how='left')
        
        # forward fill any remaining gaps after join
        merged_df = merged_df.ffill()
        
        return merged_df
    except Exception as e:
        logger.error(f"Failed to fetch step 1 data for {ticker}: {e}")
        raise
