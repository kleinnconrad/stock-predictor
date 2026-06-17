import pandas as pd
import logging

logger = logging.getLogger(__name__)

def filter_qualified_tickers(df: pd.DataFrame) -> list:
    """
    Filters the Xetra T7 dataframe to qualify German .DE retail stocks.
    
    Args:
        df (pd.DataFrame): Raw Xetra T7 DataFrame.
        
    Returns:
        list: A list of qualified ticker strings (e.g., 'SAP.DE').
    """
    logger.info("Filtering qualified tickers from Xetra T7 data.")
    
    try:
        # Assuming typical Xetra column names: 'Mnemonic', 'Instrument Status', 'Instrument Group'
        # We need to map mnemonics to Yahoo Finance .DE tickers.
        
        # Example filter logic (adjust based on actual Xetra T7 columns if different):
        # 1. Must have a Mnemonic
        if 'Mnemonic' not in df.columns:
            logger.warning("'Mnemonic' column not found. Returning empty list.")
            return []
            
        valid_df = df.dropna(subset=['Mnemonic'])
        
        # In a real scenario, you might filter by 'Instrument Group' == 'EQTY' or similar
        # For now, we take mnemonics and append '.DE'
        tickers = (valid_df['Mnemonic'] + '.DE').tolist()
        
        logger.info(f"Qualified {len(tickers)} tickers.")
        return tickers
    except Exception as e:
        logger.error(f"Failed to qualify tickers: {e}")
        return []
