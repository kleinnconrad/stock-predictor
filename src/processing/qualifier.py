import pandas as pd
import logging

logger = logging.getLogger(__name__)

def filter_qualified_tickers(df: pd.DataFrame) -> list:
    """
    Filters the Xetra T7 dataframe to qualify German .DE retail stocks.
    
    Args:
        df (pd.DataFrame): Raw Xetra T7 DataFrame.
        
    Returns:
        list: A list of dicts (e.g., [{'Ticker': 'SAP.DE', 'Company': 'SAP SE'}]).
    """
    logger.info("Filtering qualified tickers from Xetra T7 data.")
    
    try:
        # Assuming typical Xetra column names: 'Mnemonic', 'Instrument Status', 'Instrument Group'
        # We need to map mnemonics to Yahoo Finance .DE tickers.
        
        # Example filter logic (adjust based on actual Xetra T7 columns if different):
        # 1. Must have a Mnemonic
        if 'Mnemonic' not in df.columns or 'Instrument' not in df.columns:
            logger.warning("'Mnemonic' or 'Instrument' column not found. Returning empty list.")
            return []
            
        valid_df = df.dropna(subset=['Mnemonic', 'Instrument'])
        
        # Filter by Instrument Type (Common Stock) and Country/Indices
        valid_df = valid_df[valid_df['Instrument Type'] == 'CS']
        valid_groups = ['DEUTSCHLAND', 'DAX', 'MDAX', 'SDAX']
        valid_df = valid_df[valid_df['Product Assignment Group Description'].isin(valid_groups)]
        
        tickers = []
        for _, row in valid_df.iterrows():
            tickers.append({
                'Ticker': str(row['Mnemonic']) + '.DE',
                'Company': str(row['Instrument'])
            })
        
        logger.info(f"Qualified {len(tickers)} domestic German tickers.")
        return tickers
    except Exception as e:
        logger.error(f"Failed to qualify tickers: {e}")
        return []
