import yfinance as yf
import pandas as pd
import logging

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
    logger.info(f"Fetching fundamental data for {ticker}.")
    try:
        stock = yf.Ticker(ticker)
        
        # Fetching quarterly financials, balance sheet, and cashflow
        financials = stock.quarterly_financials.T
        balance_sheet = stock.quarterly_balance_sheet.T
        cashflow = stock.quarterly_cashflow.T
        
        from config.universe import FUNDAMENTAL_UNIVERSE
        
        # Combine them into one dataframe
        fundamentals = pd.concat([financials, balance_sheet, cashflow], axis=1)
        
        # Drop duplicated columns
        fundamentals = fundamentals.loc[:, ~fundamentals.columns.duplicated()]
        
        # Filter strictly by the useful variable universe
        cols_to_keep = [c for c in FUNDAMENTAL_UNIVERSE if c in fundamentals.columns]
        fundamentals = fundamentals[cols_to_keep]
        
        if fundamentals.empty:
            logger.warning(f"No fundamental data found for {ticker}.")
            return fundamentals
            
        # Ensure the index is datetime
        fundamentals.index = pd.to_datetime(fundamentals.index)
        fundamentals = fundamentals.sort_index()
        
        # Return the raw quarterly data, sorted by date (oldest to newest)
        return fundamentals
    except Exception as e:
        logger.error(f"Failed to fetch fundamental data for {ticker}: {e}")
        return pd.DataFrame()
