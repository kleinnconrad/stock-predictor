import logging
import pandas as pd
import os
from tqdm import tqdm
from ..ingestion.xetra_t7 import fetch_xetra_t7
from ..ingestion.global_macro import fetch_global_macro_universe
from ..ingestion.market_api import fetch_step1_data
from ..ingestion.funds_api import fetch_step2_fundamentals
from ..processing.qualifier import filter_qualified_tickers
from ..processing.features import engineer_features
from ..modeling.step1_macro import run_step1_model
from ..modeling.step2_funds import run_step2_model
from .json_exporter import export_prediction_json, export_diagnostics_json

logger = logging.getLogger(__name__)

def run_single(ticker: str, macro_df: pd.DataFrame = None) -> bool:
    """
    Runs the full Step 1 and Step 2 pipeline for a single ticker.
    """
    logger.info(f"Running Two-Step Cascade for single ticker: {ticker}")
    if macro_df is None:
        logger.info("Pre-caching Global Macro Universe for local execution...")
        macro_df = fetch_global_macro_universe(history_years=3)
        
    try:
        # Step 1
        merged_df = fetch_step1_data(ticker, macro_df, history_years=3)
        if merged_df.empty:
            logger.warning(f"No market data for {ticker}. Aborting.")
            return False
            
        features_df = engineer_features(merged_df)
        step1_dates, feature_diagnostics_1, pred_payload_1 = run_step1_model(features_df)
        
        export_diagnostics_json(ticker, feature_diagnostics_1, step=1)
        if pred_payload_1:
            export_prediction_json(ticker, pred_payload_1, step=1)
            
        if step1_dates.empty:
            logger.info(f"{ticker} failed Step 1. Not proceeding to Step 2.")
            return False
            
        # Step 2
        funds_df = fetch_step2_fundamentals(ticker, history_years=3)
        if funds_df.empty:
             logger.warning(f"No fundamental data for {ticker}. Aborting.")
             return False
             
        final_decision, feature_diagnostics_2, pred_payload_2 = run_step2_model(funds_df, step1_dates)
        
        export_diagnostics_json(ticker, feature_diagnostics_2, step=2)
        if final_decision:
            export_prediction_json(ticker, pred_payload_2, step=2)
            logger.info(f"*** {ticker} PASSED STEP 2 AND IS A BUY CANDIDATE! ***")
            return True
        else:
            logger.info(f"{ticker} failed Step 2.")
            return False
            
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}", exc_info=True)
        return False

def run_batch():
    """
    Runs the local execution loop over all qualified Xetra tickers.
    """
    logger.info("Starting local batch runner.")
    
    # 1. Fetch Tickers
    with open('config/settings.yaml', 'r') as f:
        settings = yaml.safe_load(f)
    raw_df = fetch_xetra_t7(settings['xetra_t7_url'])
    qualified_tickers = filter_qualified_tickers(raw_df)
    
    if not qualified_tickers:
        logger.warning("No qualified tickers found. Exiting.")
        return
        
    # 2. Cache Macro
    logger.info("Pre-caching Global Macro Universe for local execution...")
    macro_df = fetch_global_macro_universe(history_years=3)
    
    buy_candidates = []
    
    for ticker in tqdm(qualified_tickers, desc="Processing Tickers"):
        # We process sequentially in the local batch runner
        if run_single(ticker, macro_df=macro_df):
            buy_candidates.append(ticker)
            
    out_dir = os.path.join('data', 'processed')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'final_buy_signals_local.csv')
    pd.DataFrame({'ticker': buy_candidates}).to_csv(out_path, index=False)
    logger.info(f"Local batch run complete. {len(buy_candidates)} buy candidates found. Saved to {out_path}.")
