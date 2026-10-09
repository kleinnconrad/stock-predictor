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

def fetch_fred_indicators(series_ids, start_date, end_date, max_staleness_days: int) -> pd.DataFrame:
    """
    Fetches FRED series one by one so a single invalid or discontinued ID cannot
    remove every FRED indicator from the matrix.
    
    Args:
        series_ids (list): FRED series IDs.
        start_date: First observation date to request.
        end_date: Last observation date to request.
        max_staleness_days (int): Series whose latest observation is older than this
            are treated as discontinued and dropped.
        
    Returns:
        pd.DataFrame: One column per usable series, indexed by observation date.
    """
    api_key = os.getenv('FRED_API_KEY')
    frames, failed, stale = [], [], []
    for series_id in series_ids:
        try:
            raw = web.DataReader(series_id, 'fred', start_date, end_date, api_key=api_key)
        except Exception as e:
            logger.warning(f"Failed to fetch FRED series {series_id}: {e}")
            failed.append(series_id)
            continue
        series = raw[series_id].dropna() if series_id in raw.columns else pd.Series(dtype=float)
        if series.empty:
            failed.append(series_id)
            continue
        if (pd.Timestamp(end_date) - series.index.max()).days > max_staleness_days:
            stale.append(f"{series_id} (last {series.index.max().date()})")
            continue
        frames.append(series.rename(series_id))
        
    if failed:
        logger.warning(f"FRED series without data: {failed}")
    if stale:
        logger.warning(f"FRED series dropped as discontinued: {stale}")
    if not frames:
        logger.error("No FRED indicator could be fetched; the macro matrix contains Yahoo Finance data only.")
        return pd.DataFrame()
    return pd.concat(frames, axis=1, sort=True)

# Columns whose levels are already stationary (rates, spreads, indices of sentiment/stress):
# transformed with absolute differences and kept as levels.
RATE_KEYWORDS = ['TNX', 'IRX', 'VIX', 'UNRATE', 'T10Y2Y', 'EPU', 'ratio_', 'HUTTTT', 'NFCI', 'UMCSENT']
# Slow-moving economic series that additionally get a YoY acceleration (2nd derivative) feature.
MACRO_KEYWORDS = ['CPIAUCSL', 'CP00MI15', 'M2SL', 'PAYEMS', 'UNRATE', 'WALCL', 'ASSETS', 'PERMIT', 'ICSA', 'DGORDER']
# Stress indicators that get a rolling 2-year Z-score (regime normalization).
ZSCORE_KEYWORDS = ['VIX', 'credit_spread', 'EPU']
MOMENTUM_WINDOWS = [21, 63, 126, 252]
ZSCORE_WINDOW = 504

def engineer_macro_features(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands the aligned macro levels (Yahoo Finance and FRED alike) into stationary features.
    
    Rate-like columns keep their level and use absolute differences; all other columns use
    percentage changes and never expose their non-stationary level. Every column gets
    multi-timeframe momentum and distance to its 200-day SMA; slow-moving economic series
    get a YoY acceleration feature and stress indicators a rolling 2-year Z-score.
    
    Args:
        macro_df (pd.DataFrame): Business-day aligned macro levels.
        
    Returns:
        pd.DataFrame: The expanded feature matrix.
    """
    logger.info("Applying quantitative feature engineering to the macro cache...")
    macro_df = macro_df.copy()
    
    # A. Interaction Ratios
    def safe_ratio(num, den, col_name):
        if num in macro_df.columns and den in macro_df.columns:
            macro_df[col_name] = macro_df[num] / macro_df[den]

    safe_ratio('HG=F', 'GC=F', 'ratio_copper_gold')
    safe_ratio('HYG', 'LQD', 'ratio_credit_spread')
    safe_ratio('XLY', 'XLP', 'ratio_consumer_risk')
    safe_ratio('SPY', 'TLT', 'ratio_risk_on_off')
    safe_ratio('XLK', 'SPY', 'ratio_tech_dominance')
    safe_ratio('IGOV', 'TLT', 'ratio_intl_vs_us_bonds')
    
    col_dict = {}
    for col in macro_df.columns:
        series = macro_df[col]
        is_rate_or_spread = any(kw in col for kw in RATE_KEYWORDS)
        
        # 0. Preserve Stationary Levels
        if is_rate_or_spread:
            col_dict[f'{col}_Level'] = series
            
        # 1. Multi-Timeframe Momentum
        for w in MOMENTUM_WINDOWS:
            if is_rate_or_spread:
                col_dict[f'{col}_{w}D_diff'] = series.diff(w)
            else:
                col_dict[f'{col}_{w}D_ret'] = series.pct_change(w, fill_method=None)
                
        # 2. Distance to Trend (200-day SMA)
        sma_200 = series.rolling(window=200).mean()
        if is_rate_or_spread:
            col_dict[f'{col}_Dist_SMA200'] = series - sma_200
        else:
            col_dict[f'{col}_Dist_SMA200'] = (series / sma_200) - 1.0
            
        # 3. Macro Acceleration (2nd Derivative): YoY change now vs. 3 months ago
        if any(kw in col for kw in MACRO_KEYWORDS):
            if is_rate_or_spread:
                current_1Y_change = series.diff(252)
                past_1Y_change = series.shift(63).diff(252)
            else:
                current_1Y_change = series.pct_change(252, fill_method=None)
                past_1Y_change = series.shift(63).pct_change(252, fill_method=None)
            col_dict[f'{col}_YoY_Accel_3M'] = current_1Y_change - past_1Y_change
            
        # 4. Rolling 2-Year Z-Scores
        if any(kw in col for kw in ZSCORE_KEYWORDS):
            roll_mean = series.rolling(window=ZSCORE_WINDOW).mean()
            roll_std = series.rolling(window=ZSCORE_WINDOW).std() + 1e-8
            col_dict[f'{col}_Roll_ZScore_2Y'] = (series - roll_mean) / roll_std
            
    expanded_macro = pd.DataFrame(col_dict, index=macro_df.index)
    
    # Clean infinities caused by ratio divisions
    expanded_macro = expanded_macro.replace([np.inf, -np.inf], np.nan)
    
    # Drop columns that ended up being completely NaN
    expanded_macro = expanded_macro.dropna(axis=1, how='all')
    return expanded_macro.ffill()

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
    settings = load_settings()
    lag_days = settings['fred_publication_lag_days']
    fred_start = start_date - datetime.timedelta(days=max(lag_days.values()) + 31)
    fred_df = fetch_fred_indicators(
        ALL_FRED_INDICATORS, fred_start, end_date,
        max_staleness_days=int(settings['fred_max_staleness_days']),
    )
    fred_df = apply_publication_lag(fred_df, lag_days)
        
    # 3. Merge, forward-fill and align to business days (up to yesterday)
    global_macro_df = yf_df.join(fred_df, how='outer')
    global_macro_df.index = pd.to_datetime(global_macro_df.index)
    last_closed_day = pd.Timestamp(current_german_date) - pd.Timedelta(days=1)
    global_macro_df = align_to_business_days(global_macro_df, end_date=last_closed_day)
    global_macro_df = global_macro_df.dropna(how='all')
    
    expanded_macro = engineer_macro_features(global_macro_df)
    logger.info(f"Global macro engineering complete. Expanded Matrix Shape: {expanded_macro.shape}")
    return expanded_macro
