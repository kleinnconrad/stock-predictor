import logging
import pandas as pd
import datetime
import os
import yaml
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

def load_settings(config_path: str = 'config/settings.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_batch():
    logger.info("Starting batch runner.")
    
    settings = load_settings()
    
    horizon_days = settings['horizon_days']
    threshold = settings['threshold']
    step1_history_years = settings['step1_history_years']
    step2_history_years = settings['step2_history_years']
    n_features = settings['features_to_select']
    
    today = datetime.date.today()
    
    # --- Data Ingestion: Qualified Tickers ---
    xetra_df = fetch_xetra_t7(settings['xetra_t7_url'])
    qualified_tickers = filter_qualified_tickers(xetra_df)
    
    if not qualified_tickers:
        logger.warning("No qualified tickers found. Exiting.")
        return
        
    # --- Data Ingestion: Step 1 (Global Macro Caching) ---
    logger.info("Fetching global macro universe (Caching ONCE to prevent YF API Bans).")
    global_macro_df = fetch_global_macro_universe(step1_history_years)
    
    start_date_step1 = today - datetime.timedelta(days=365 * step1_history_years)
    buy_candidates = []
    
    # Process each ticker with tqdm
    for ticker in tqdm(qualified_tickers, desc="Processing Tickers"):
        logger.info(f"Processing ticker: {ticker}")
        
        try:
            # Step 1: Market Data + Cached Macro Data
            step1_df = fetch_step1_data(ticker, global_macro_df, start_date_step1, today)
            if step1_df.empty:
                continue
                
            # Engineer features for step 1 (including Target variable)
            step1_df = engineer_features(step1_df, horizon_days, threshold)
            
            # Execute Step 1 Model
            step1_metrics, step1_filtered_df = execute_step1(step1_df, n_features)
            
            if step1_metrics.get("predicted_class") != "UP":
                logger.info(f"{ticker} failed Step 1 criteria.")
                continue
                
            # --- Data Ingestion: Step 2 (Fundamentals) ---
            funds_df = fetch_fundamentals(ticker)
            
            if funds_df.empty:
                 logger.warning(f"No fundamental data for {ticker}, skipping Step 2.")
                 continue
                 
            start_date_step2 = today - datetime.timedelta(days=365 * step2_history_years)
            funds_df = funds_df[funds_df.index >= pd.to_datetime(start_date_step2)]
            
            # Join fundamentals to the filtered Step 1 DF
            step2_df = step1_filtered_df.join(funds_df, how='inner')
            
            # Execute Step 2 Model
            step2_metrics, final_prediction = execute_step2(step2_df, n_features)
            
            # JSON Payload assembly
            payload = {
                "stock_name": ticker,
                "prediction_date": str(today),
                "applied_parameters": {
                    "horizon_days": horizon_days,
                    "threshold": threshold,
                    "step1_history_years": step1_history_years,
                    "step2_history_years": step2_history_years,
                    "features_to_select": n_features
                },
                "step1_model": step1_metrics,
                "step2_model": step2_metrics,
                "final_prediction": final_prediction
            }
            
            # Extract feature diagnostics and export separately
            diagnostics_payload = {
                "stock_name": ticker,
                "prediction_date": str(today),
                "step1": step1_metrics.pop("feature_diagnostics", {}),
                "step2": step2_metrics.pop("feature_diagnostics", {})
            }
            
            export_prediction_json(ticker, payload)
            export_feature_diagnostics_json(ticker, diagnostics_payload)
            
            if final_prediction == "UP":
                buy_candidates.append(ticker)
                
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}", exc_info=True)
            
    # Save Buy Candidates
    out_dir = os.path.join('data', 'processed')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'final_buy_signals.csv')
    pd.DataFrame({'ticker': buy_candidates}).to_csv(out_path, index=False)
    logger.info(f"Batch run complete. {len(buy_candidates)} buy candidates found.")
