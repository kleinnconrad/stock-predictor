import os
import sys

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import random
import argparse
import yaml
import datetime
import pandas as pd
from tqdm import tqdm
from src.ingestion.global_macro import fetch_global_macro_universe
from src.ingestion.market_api import fetch_step1_data
from src.processing.features import engineer_features
from src.modeling.step1_macro import execute_step1
from src.orchestration.json_exporter import export_prediction_json, export_feature_diagnostics_json

def apply_anti_jitter():
    """
    Sleeps for a random interval to prevent Yahoo Finance from banning the IP.
    Critical when running parallel matrix runners on GitHub Actions.
    """
    jitter = random.uniform(0.5, 2.5)
    time.sleep(jitter)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0, help="Matrix shard index")
    parser.add_argument("--total", type=int, default=1, help="Total number of runners")
    args = parser.parse_args()

    os.makedirs('data/state', exist_ok=True)
    
    if not os.path.exists('data/state/qualified_tickers.csv'):
        raise FileNotFoundError("qualified_tickers.csv missing! Did Action 1 run and persist the artifact?")
        
    all_tickers = pd.read_csv('data/state/qualified_tickers.csv')['Ticker'].tolist()
    all_tickers.sort() # Ensure deterministic ordering
    
    # Mathematical Sharding for Distributed GitHub Actions
    shard_tickers = all_tickers[args.shard :: args.total]
    print(f"[Runner {args.shard}/{args.total}] Processing {len(shard_tickers)} tickers.")
    
    print("Pre-caching Global Macro Universe...")
    macro_df = fetch_global_macro_universe(history_years=3)
    
    passed_tickers = []
    step1_dates_dict = {}
    
    for ticker in tqdm(shard_tickers, desc=f"Step 1 - Shard {args.shard}"):
        apply_anti_jitter()
        try:
            merged_df = fetch_step1_data(ticker, macro_df, history_years=3)
            if merged_df.empty:
                continue
                
            features_df = engineer_features(merged_df)
            
            metrics, filtered_df = execute_step1(features_df)
            step1_dates = filtered_df.index
            feature_diagnostics = metrics.pop('feature_diagnostics', {}) if metrics else {}
            
            with open('config/settings.yaml', 'r') as f:
                settings = yaml.safe_load(f)
                
            pred_payload = {
                "stock_name": ticker,
                "prediction_date": str(datetime.date.today()),
                "applied_parameters": {
                    "horizon_days": settings.get('horizon_days', 126),
                    "threshold": settings.get('threshold', 0.15),
                    "step1_history_years": settings.get('step1_history_years', 10),
                    "step2_history_years": settings.get('step2_history_years', 4),
                    "features_to_select": settings.get('features_to_select', 12)
                },
                "step1_model": metrics,
                "final_prediction": "PENDING"
            } if metrics else {}
            
            # Export JSON payloads
            export_feature_diagnostics_json(ticker, feature_diagnostics)
            if pred_payload:
                latest_is_up = (metrics.get('predicted_class') == 'UP')
                pred_payload["final_prediction"] = "UP" if latest_is_up else "NOT_UP"
                export_prediction_json(ticker, pred_payload)
                
            if metrics.get('predicted_class') == 'UP':
                passed_tickers.append(ticker)
                # Store valid dates for Step 2 execution
                step1_dates_dict[ticker] = [str(d) for d in step1_dates]
                
        except Exception as e:
            print(f"Failed {ticker} in Step 1: {e}")
            
    # Save State Artifacts for Step 2
    output_path = f'data/state/step1_passed_shard_{args.shard}.csv'
    pd.DataFrame({'Ticker': passed_tickers}).to_csv(output_path, index=False)
    
    json_path = f'data/state/step1_dates_shard_{args.shard}.json'
    with open(json_path, 'w') as f:
        json.dump(step1_dates_dict, f)
        
    print(f"[Runner {args.shard}] Saved {len(passed_tickers)} passing tickers.")

if __name__ == "__main__":
    main()
