import pandas as pd
import yfinance as yf
import datetime
import logging

logger = logging.getLogger(__name__)

def fetch_step1_data(ticker: str, global_macro_df: pd.DataFrame, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """
    Fetches target stock OHLCV data and merges it with the global macro cache.
    
    Args:
        ticker (str): The stock ticker (e.g., 'SAP.DE').
        global_macro_df (pd.DataFrame): Pre-cached global macro dataset.
        start_date (datetime.date): Start date.
        end_date (datetime.date): End date.
        
    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    logger.info(f"Fetching market pricing for target {ticker}.")
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            logger.warning(f"No pricing data found for {ticker}.")
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
