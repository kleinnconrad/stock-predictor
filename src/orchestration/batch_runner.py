import logging
import pandas as pd
import os
from tqdm import tqdm
from ..ingestion.xetra_t7 import fetch_xetra_t7
from ..ingestion.global_macro import fetch_global_macro_universe
from ..processing.qualifier import filter_qualified_tickers
from .steps import history_window, run_step1_for_ticker, run_step2_for_ticker
from config.settings import load_settings

logger = logging.getLogger(__name__)

def run_single(ticker_data: dict, macro_df: pd.DataFrame = None) -> bool:
    """
    Runs the full Step 1 and Step 2 pipeline for a single ticker.
    
    Uses the same per-ticker runners as the GitHub Actions scripts, so local runs
    produce identical payloads and diagnostics.
    
    Args:
        ticker_data (dict): {'Ticker': ..., 'Company': ...}.
        macro_df (pd.DataFrame): Pre-cached macro matrix (fetched if omitted).
        
    Returns:
        bool: True if the ticker is a final buy candidate.
    """
    ticker = ticker_data['Ticker']
    company_name = ticker_data['Company']
    logger.info(f"Running Two-Step Cascade for single ticker: {ticker} ({company_name})")
    settings = load_settings()
    if macro_df is None:
        logger.info("Pre-caching Global Macro Universe for local execution...")
        _, fetch_years = history_window(settings)
        macro_df = fetch_global_macro_universe(history_years=fetch_years)
        
    try:
        payload = run_step1_for_ticker(ticker, company_name, macro_df, settings)
        if payload is None:
            return False
        if payload['final_prediction'] != 'UP':
            logger.info(f"{ticker} failed Step 1 (cv_status: {payload['step1_model'].get('cv_status')}). Not proceeding to Step 2.")
            return False
            
        payload = run_step2_for_ticker(ticker)
        return payload['final_prediction'] == 'UP_FINAL_BUY'
    except Exception:
        logger.exception(f"Error processing {ticker}")
        return False

def run_batch():
    """
    Runs the local execution loop over all qualified Xetra tickers.
    """
    logger.info("Starting local batch runner.")
    
    # 1. Fetch Tickers
    settings = load_settings()
    raw_df = fetch_xetra_t7(settings['xetra_t7_url'])
    qualified_tickers = filter_qualified_tickers(raw_df)
    
    if not qualified_tickers:
        logger.warning("No qualified tickers found. Exiting.")
        return
        
    # 2. Cache Macro
    logger.info("Pre-caching Global Macro Universe for local execution...")
    _, fetch_years = history_window(settings)
    macro_df = fetch_global_macro_universe(history_years=fetch_years)
    
    buy_candidates = []
    
    for ticker_data in tqdm(qualified_tickers, desc="Processing Tickers"):
        # We process sequentially in the local batch runner
        if run_single(ticker_data, macro_df=macro_df):
            buy_candidates.append(ticker_data['Ticker'])
            
    out_dir = os.path.join('data', 'processed')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'final_buy_signals_local.csv')
    pd.DataFrame({'ticker': buy_candidates}).to_csv(out_path, index=False)
    logger.info(f"Local batch run complete. {len(buy_candidates)} buy candidates found. Saved to {out_path}.")
    
    import json
    import glob
    import time
    from datetime import datetime
    
    payload_dict = {}
    for f_path in glob.glob("outputs/predictions/*.json"):
        try:
            with open(f_path, "r") as file:
                data = json.load(file)
                ticker = data.get("stock_name")
                if ticker:
                    payload_dict[ticker] = data
        except Exception as e:
            logger.error(f"Failed to read {f_path}: {e}")
            
    report = {
        "execution_date": datetime.utcnow().isoformat() + "Z",
        "parameters": settings,
        "predictions": list(payload_dict.values())
    }
    
    with open(os.path.join(out_dir, 'full_batch_report.json'), "w") as out:
        json.dump(report, out, indent=2)
        
    history_dir = os.path.join(out_dir, "history")
    today_dt = datetime.utcnow()
    month_str = today_dt.strftime("%Y-%m")
    today_str = today_dt.strftime("%Y-%m-%d")
    month_dir = os.path.join(history_dir, month_str)
    os.makedirs(month_dir, exist_ok=True)
    
    hist_file = os.path.join(month_dir, f"report_{today_str}.json")
    with open(hist_file, "w") as out:
        json.dump(report, out, indent=2)
        
    now = time.time()
    for f_path in glob.glob(os.path.join(history_dir, "*", "report_*.json")):
        if os.stat(f_path).st_mtime < now - 180 * 86400:
            os.remove(f_path)
            
    try:
        from ..processing.uplift_evaluator import evaluate_uplift
        evaluate_uplift()
    except Exception as e:
        logger.error(f"Failed to run uplift evaluation: {e}")
