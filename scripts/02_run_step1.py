import os
import json
import time
import random
import argparse
import pandas as pd
from tqdm import tqdm
from src.ingestion.global_macro import fetch_global_macro_universe
from src.ingestion.market_api import fetch_step1_data
from src.processing.features import engineer_features
from src.modeling.step1_macro import run_step1_model
from src.orchestration.json_exporter import export_prediction_json, export_diagnostics_json

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
            
            step1_dates, feature_diagnostics, pred_payload = run_step1_model(features_df)
            
            if not step1_dates.empty:
                passed_tickers.append(ticker)
                # Store valid dates for Step 2 execution
                step1_dates_dict[ticker] = [str(d) for d in step1_dates]
                
            # Export JSON payloads
            export_diagnostics_json(ticker, feature_diagnostics, step=1)
            if pred_payload:
                export_prediction_json(ticker, pred_payload, step=1)
                
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
