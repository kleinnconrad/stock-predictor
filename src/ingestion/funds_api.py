import yfinance as yf
import pandas as pd
import logging
import time
import random

logger = logging.getLogger(__name__)

def fetch_fundamentals(ticker: str) -> pd.DataFrame:
    """
    Fetches fundamental financial statement data from Yahoo Finance and creates a 
    time-series dataframe forward-filling quarterly/annual data to daily.
    
    Args:
        ticker (str): The stock ticker symbol.
        
    Returns:
        pd.DataFrame: A time-indexed DataFrame containing fundamentals, forward-filled.
    """
    tickers_to_try = [ticker]
    if ticker.endswith('.DE'):
        # Safely try the Frankfurt exchange if Xetra fails
        tickers_to_try.append(ticker[:-3] + '.F')
        
    from config.universe import FUNDAMENTAL_UNIVERSE
        
    for t in tickers_to_try:
        # Introduce a jitter of 0.5 to 1.5 seconds to avoid aggressive rate limiting
        jitter = random.uniform(0.5, 1.5)
        time.sleep(jitter)
        
        logger.info(f"Fetching fundamental data for {t}...")
        try:
            stock = yf.Ticker(t)
            
            # Fetching quarterly financials, balance sheet, and cashflow
            financials = stock.quarterly_financials.T
            balance_sheet = stock.quarterly_balance_sheet.T
            cashflow = stock.quarterly_cashflow.T
            
            # Combine them into one dataframe
            fundamentals = pd.concat([financials, balance_sheet, cashflow], axis=1)
            
            # Drop duplicated columns
            fundamentals = fundamentals.loc[:, ~fundamentals.columns.duplicated()]
            
            # Filter strictly by the useful variable universe
            cols_to_keep = [c for c in FUNDAMENTAL_UNIVERSE if c in fundamentals.columns]
            fundamentals = fundamentals[cols_to_keep]
            
            if fundamentals.empty:
                logger.warning(f"No fundamental data found for {t}.")
                continue
                
            # Ensure the index is datetime
            fundamentals.index = pd.to_datetime(fundamentals.index)
            fundamentals = fundamentals.sort_index()
            
            # Return the raw quarterly data, sorted by date (oldest to newest)
            return fundamentals
            
        except Exception as e:
            logger.warning(f"Failed to fetch fundamental data for {t}: {e}")
            
    logger.error(f"Exhausted all ticker variations for {ticker}. Returning empty DataFrame.")
    return pd.DataFrame()
