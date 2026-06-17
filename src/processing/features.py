import pandas as pd
import logging

logger = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame, horizon_days: int = 126, threshold: float = 0.15) -> pd.DataFrame:
    """
    Engineers technical indicators, macroeconomic features, and the target variable.
    
    Args:
        df (pd.DataFrame): The combined pricing and macro/fundamental DataFrame.
        horizon_days (int): The prediction horizon in days.
        threshold (float): The target threshold for positive classification (e.g., 0.15 for 15%).
        
    Returns:
        pd.DataFrame: DataFrame with engineered features and the 'Target' column.
    """
    logger.info(f"Engineering features with horizon={horizon_days} and threshold={threshold}")
    
    if df.empty or 'Close' not in df.columns:
        logger.warning("DataFrame is empty or missing 'Close' price. Cannot engineer features.")
        return df

    try:
        # Create a copy to avoid SettingWithCopyWarning
        data = df.copy()
        
        # Momentum / Returns (1 Month, 1 Quarter, 6 Months, 1 Year)
        data['Ret_21D'] = data['Close'].pct_change(periods=21)
        data['Ret_63D'] = data['Close'].pct_change(periods=63)
        data['Ret_126D'] = data['Close'].pct_change(periods=126)
        data['Ret_252D'] = data['Close'].pct_change(periods=252)
        
        # Moving Average Distances
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['Dist_SMA_50'] = (data['Close'] - data['SMA_50']) / data['SMA_50']
        
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        data['Dist_SMA_200'] = (data['Close'] - data['SMA_200']) / data['SMA_200']
        
        # Volatility
        data['Vol_21D'] = data['Close'].pct_change().rolling(window=21).std()
        
        # 2. Engineering the Target Variable
        # Calculate future return over horizon_days
        data['Future_Return'] = data['Close'].shift(-horizon_days) / data['Close'] - 1.0
        
        # Target Class: 1 if Future_Return >= threshold, else 0
        data['Target'] = (data['Future_Return'] >= threshold).astype(int)
        
        # The 'Target' for the most recent `horizon_days` will be NaN (since we can't look into the future).
        # We must drop rows where Target calculation is impossible for training, 
        # BUT we need the most recent rows for *prediction*.
        # So we keep Future_Return as NaN, and we'll separate training and prediction sets later.
        data.loc[data['Future_Return'].isna(), 'Target'] = pd.NA
        
        # Drop intermediary columns if desired, but SMA_50 etc might be useful
        data = data.drop(columns=['SMA_50', 'SMA_200'])
        
        return data
    except Exception as e:
        logger.error(f"Error engineering features: {e}")
        raise
