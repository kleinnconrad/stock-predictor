import logging
import pandas as pd
import os
from tqdm import tqdm
from ..ingestion.xetra_t7 import fetch_xetra_t7
from ..ingestion.global_macro import fetch_global_macro_universe
from ..ingestion.market_api import fetch_step1_data
from ..ingestion.funds_api import fetch_fundamentals
from ..processing.qualifier import filter_qualified_tickers
from ..processing.features import engineer_features
from ..modeling.step1_macro import execute_step1
from ..modeling.step2_funds import execute_step2
from .json_exporter import export_prediction_json, export_feature_diagnostics_json

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
        metrics_1, filtered_df_1 = execute_step1(features_df, ticker=ticker)
        step1_dates = filtered_df_1.index
        feature_diagnostics_1 = metrics_1.pop('feature_diagnostics', {}) if metrics_1 else {}
        pred_payload_1 = metrics_1
        
        combined_diagnostics = {"step1_macro": feature_diagnostics_1}
        export_feature_diagnostics_json(ticker, combined_diagnostics)
        
        # We will export the combined payload at the end. For now, export Step 1 alone.
        combined_payload = {"step1_model": pred_payload_1, "final_prediction": "PENDING"}
        if pred_payload_1:
            export_prediction_json(ticker, combined_payload)
            
        latest_macro_pred = metrics_1.get("predicted_class", "NOT_UP")
        if latest_macro_pred != "UP":
            logger.info(f"{ticker} failed Step 1 macro threshold (Predicted: {latest_macro_pred}). Not proceeding to Step 2.")
            combined_payload["final_prediction"] = "NOT_UP"
            export_prediction_json(ticker, combined_payload)
            return False
            
        # Step 2
        funds_df = fetch_fundamentals(ticker)
        
        mat_dir = os.path.join('outputs', 'matrices')
        os.makedirs(mat_dir, exist_ok=True)
        funds_df.to_csv(os.path.join(mat_dir, f"{ticker}_step2_funds.csv"))
        
        if funds_df.empty:
             logger.warning(f"No fundamental data for {ticker}. Aborting.")
             combined_payload["final_prediction"] = "UP"
             export_prediction_json(ticker, combined_payload)
             return False
             
        metrics_2, latest_pred_class = execute_step2(funds_df)
        feature_diagnostics_2 = metrics_2.pop('feature_diagnostics', {}) if metrics_2 else {}
        pred_payload_2 = metrics_2
        final_decision = latest_pred_class == "UP"
        
        combined_diagnostics["step2_funds"] = feature_diagnostics_2
        export_feature_diagnostics_json(ticker, combined_diagnostics)
        
        combined_payload["step2_model"] = pred_payload_2
        combined_payload["final_prediction"] = "UP_FINAL_BUY" if final_decision else "UP"
        
        # Unconditionally export the final updated payload regardless of outcome
        export_prediction_json(ticker, combined_payload)
        
        if final_decision:
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
