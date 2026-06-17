import pandas as pd
import logging
import requests
import os

logger = logging.getLogger(__name__)

def fetch_xetra_t7(url: str) -> pd.DataFrame:
    """
    Downloads and parses the Xetra T7 dump to identify all tradable German stocks.
    
    Args:
        url (str): The URL to the Xetra T7 CSV data.
        
    Returns:
        pd.DataFrame: DataFrame containing the raw Xetra T7 data.
    """
    logger.info(f"Initiating direct download from: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        csv_response = requests.get(url, headers=headers, timeout=30)
        csv_response.raise_for_status()
        
        # Save to data directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(project_root, 'data', 'raw')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 't7-xetr-allTradableInstruments.csv')
        
        with open(output_path, 'wb') as f:
            f.write(csv_response.content)
            
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Success! Downloaded {file_size_mb:.2f} MB to {output_path}")
        
        # Now read it with pandas
        df = pd.read_csv(output_path, sep=';', skiprows=2, on_bad_lines='skip')
        logger.info(f"Successfully loaded {len(df)} rows from Xetra T7 dump.")
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch Xetra T7 dump: {e}")
        raise
