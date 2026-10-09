import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import datetime
import logging
import os
from typing import Dict
from config.universe import ALL_YF_TICKERS, ALL_FRED_INDICATORS
from config.settings import load_settings

logger = logging.getLogger(__name__)

def infer_observation_frequency(index: pd.DatetimeIndex) -> str:
    """
    Infers the release frequency of a series from the spacing of its observation dates.
    
    Args:
        index (pd.DatetimeIndex): Observation dates of a series (NaNs already removed).
        
    Returns:
        str: One of 'daily', 'weekly', 'monthly' or 'quarterly'.
    """
    if len(index) < 2:
        return 'monthly'
    median_gap_days = pd.Series(index).diff().dt.days.median()
    if median_gap_days <= 3:
        return 'daily'
    if median_gap_days <= 10:
        return 'weekly'
    if median_gap_days <= 45:
        return 'monthly'
    return 'quarterly'

def apply_publication_lag(fred_df: pd.DataFrame, lag_days: Dict[str, int]) -> pd.DataFrame:
    """
    Moves every FRED observation to the date it was (conservatively) available.
    
    FRED dates an observation by the start of the period it covers, not by its release:
    September CPI is dated September 1 but published mid-October. Without this shift the
    training rows contain information that was not public on that date.
    
    Args:
        fred_df (pd.DataFrame): FRED series indexed by observation date.
        lag_days (Dict[str, int]): Publication lag in calendar days per frequency.
        
    Returns:
        pd.DataFrame: The same series, each indexed by its assumed publication date.
    """
    shifted = []
    for col in fred_df.columns:
        series = fred_df[col].dropna()
        if series.empty:
            continue
        frequency = infer_observation_frequency(series.index)
        series.index = series.index + pd.Timedelta(days=int(lag_days[frequency]))
        shifted.append(series)
    if not shifted:
        return pd.DataFrame()
    return pd.concat(shifted, axis=1, sort=True)

def align_to_business_days(df: pd.DataFrame, end_date) -> pd.DataFrame:
    """
    Forward-fills every column across all observation dates, then samples business days.
    
    `resample('B').ffill()` only reindexes rows: a value dated on a weekend (e.g. weekly
    FRED series dated Saturdays) or a column gap caused by a market holiday is never
    carried forward, which leaves large parts of the matrix empty.
    
    Args:
        df (pd.DataFrame): Merged series indexed by observation date (may include weekends).
        end_date: Last business day to include.
        
    Returns:
        pd.DataFrame: One row per business day with the latest known value of every column.
    """
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='last')]
    df = df.ffill()
    business_days = pd.bdate_range(df.index.min(), pd.Timestamp(end_date))
    return df.reindex(business_days, method='ffill')

def fetch_global_macro_universe(history_years: int) -> pd.DataFrame:
    """
    Downloads the full 360-degree macroeconomic universe from YF and FRED,
    and engineers an expanded quantitative feature matrix.
    
    Args:
        history_years (int): Number of years of history to fetch.
        
    Returns:
        pd.DataFrame: A heavily expanded, pre-cached, stationary global macroeconomic dataframe.
    """
    # Use Europe/Berlin time to ensure consistency between local and cloud runs
    current_german_date = pd.Timestamp.now(tz='Europe/Berlin').date()
    start_date = current_german_date - datetime.timedelta(days=365 * history_years)
    end_date = current_german_date
    
    logger.info(f"Fetching {len(ALL_YF_TICKERS)} YF tickers and {len(ALL_FRED_INDICATORS)} FRED indicators for Global Macro Cache.")
    
    # 1. Fetch Yahoo Finance macro indices
    yf_df = yf.download(ALL_YF_TICKERS, start=start_date, end=end_date, progress=False)
    
    # Explicitly strip intraday/live data to stabilize the model
    yf_df = yf_df[yf_df.index.date < current_german_date]
    
    if 'Adj Close' in yf_df.columns:
        yf_df = yf_df['Adj Close']
    elif 'Close' in yf_df.columns:
        yf_df = yf_df['Close']
        
    if isinstance(yf_df.columns, pd.MultiIndex):
        yf_df.columns = yf_df.columns.droplevel('Ticker')
        
    # 2. Fetch FRED indicators and shift them to their publication dates
    lag_days = load_settings()['fred_publication_lag_days']
    fred_start = start_date - datetime.timedelta(days=max(lag_days.values()) + 31)
    try:
        api_key = os.getenv('FRED_API_KEY')
        fred_df = web.DataReader(ALL_FRED_INDICATORS, 'fred', fred_start, end_date, api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to fetch FRED indicators: {e}")
        fred_df = pd.DataFrame()
    fred_df = apply_publication_lag(fred_df, lag_days)
        
    # 3. Merge, forward-fill and align to business days (up to yesterday)
    global_macro_df = yf_df.join(fred_df, how='outer')
    global_macro_df.index = pd.to_datetime(global_macro_df.index)
    last_closed_day = pd.Timestamp(current_german_date) - pd.Timedelta(days=1)
    global_macro_df = align_to_business_days(global_macro_df, end_date=last_closed_day)
    global_macro_df = global_macro_df.dropna(how='all')
    
    # =========================================================================
    # ADVANCED QUANTITATIVE FEATURE ENGINEERING
    # =========================================================================
    logger.info("Applying massive quantitative feature engineering matrix to macro cache...")
    
    # A. Interaction Ratios
    def safe_ratio(num, den, col_name):
        if num in global_macro_df.columns and den in global_macro_df.columns:
            global_macro_df[col_name] = global_macro_df[num] / global_macro_df[den]

    safe_ratio('HG=F', 'GC=F', 'ratio_copper_gold')
    safe_ratio('HYG', 'LQD', 'ratio_credit_spread')
    safe_ratio('XLY', 'XLP', 'ratio_consumer_risk')
    safe_ratio('SPY', 'TLT', 'ratio_risk_on_off')
    safe_ratio('XLK', 'SPY', 'ratio_tech_dominance')
    safe_ratio('IGOV', 'TLT', 'ratio_intl_vs_us_bonds')
    
    # B. Timeframes and Rate Keywords
    windows = [21, 63, 126, 252] 
    RATE_KEYWORDS = ['TNX', 'IRX', 'VIX', 'UNRATE', 'T10Y2Y', 'EPU', 'ratio_', 'HUTTTT', 'NFCI', 'BAML', 'UMCSENT']
    MACRO_KEYWORDS = ['CPIAUCSL', 'M2SL', 'PAYEMS', 'UNRATE', 'ASSETS', 'PROIND', 'PERMIT']
    
    col_dict = {}
    
    for col in global_macro_df.columns:
        if col in ALL_FRED_INDICATORS:
            # FRED indicators are updated infrequently (e.g. monthly/quarterly).
            # We only keep their forward-filled levels and a rolling Z-score.
            # (Forward fill is already applied globally via resample('B').ffill())
            col_dict[f'{col}_Level'] = global_macro_df[col]
            
            # Z-scale using a rolling 2-Year window to prevent look-ahead bias
            roll_mean = global_macro_df[col].rolling(window=504).mean()
            roll_std = global_macro_df[col].rolling(window=504).std() + 1e-8
            col_dict[f'{col}_ZScore'] = (global_macro_df[col] - roll_mean) / roll_std
            continue

        is_rate_or_spread = any(kw in col for kw in RATE_KEYWORDS)
        
        # 0. Preserve Stationary Levels
        if is_rate_or_spread:
            col_dict[f'{col}_Level'] = global_macro_df[col]
            
        # 1. Multi-Timeframe Momentum
        for w in windows:
            if is_rate_or_spread:
                col_dict[f'{col}_{w}D_diff'] = global_macro_df[col].diff(w)
            else:
                col_dict[f'{col}_{w}D_ret'] = global_macro_df[col].pct_change(w, fill_method=None)
                
        # 2. Distance to Trend (200-day SMA)
        sma_200 = global_macro_df[col].rolling(window=200).mean()
        if is_rate_or_spread:
            col_dict[f'{col}_Dist_SMA200'] = global_macro_df[col] - sma_200
        else:
            col_dict[f'{col}_Dist_SMA200'] = (global_macro_df[col] / sma_200) - 1.0
            
        # 3. Macro Acceleration (2nd Derivative)
        if any(kw in col for kw in MACRO_KEYWORDS):
            if is_rate_or_spread:
                current_1Y_change = global_macro_df[col].diff(252)
                past_1Y_change = global_macro_df[col].shift(63).diff(252)
            else:
                current_1Y_change = global_macro_df[col].pct_change(252, fill_method=None)
                past_1Y_change = global_macro_df[col].shift(63).pct_change(252, fill_method=None)
                
            col_dict[f'{col}_YoY_Accel_3M'] = current_1Y_change - past_1Y_change
            
        # 4. Rolling 2-Year Z-Scores
        if 'VIX' in col or 'credit_spread' in col or 'EPU' in col:
            roll_mean = global_macro_df[col].rolling(window=504).mean()
            roll_std = global_macro_df[col].rolling(window=504).std() + 1e-8
            col_dict[f'{col}_Roll_ZScore_2Y'] = (global_macro_df[col] - roll_mean) / roll_std
            
    # Assemble expanded matrix
    expanded_macro = pd.DataFrame(col_dict, index=global_macro_df.index)
    
    # Clean infinities caused by ratio divisions
    expanded_macro = expanded_macro.replace([np.inf, -np.inf], np.nan)
    
    # Drop columns that ended up being completely NaN
    expanded_macro = expanded_macro.dropna(axis=1, how='all')
    expanded_macro = expanded_macro.ffill()
    
    logger.info(f"Global macro engineering complete. Expanded Matrix Shape: {expanded_macro.shape}")
    return expanded_macro
