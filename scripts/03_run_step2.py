import os
import sys

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import random
import glob
import argparse
import pandas as pd
from tqdm import tqdm
from src.ingestion.funds_api import fetch_fundamentals
from src.modeling.step2_funds import execute_step2
from src.orchestration.json_exporter import export_prediction_json, export_feature_diagnostics_json

def apply_anti_jitter():
    """
    Random delay to prevent API bans across concurrent nodes.
    """
    jitter = random.uniform(0.5, 2.5)
    time.sleep(jitter)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0, help="Matrix shard index")
    parser.add_argument("--total", type=int, default=1, help="Total number of runners")
    args = parser.parse_args()

    os.makedirs('data/processed', exist_ok=True)
    
    # 1. Consolidate passing tickers from Action 2 Artifacts
    shard_csvs = glob.glob('data/state/step1_passed_shard_*.csv')
    if not shard_csvs:
        raise FileNotFoundError("No Step 1 CSV artifacts found! Did Action 2 run successfully?")
        
    all_passed_tickers = []
    for f in shard_csvs:
        df = pd.read_csv(f)
        all_passed_tickers.extend(df['Ticker'].tolist())
        
    all_passed_tickers = list(set(all_passed_tickers)) # De-duplicate
    all_passed_tickers.sort() # Ensure deterministic ordering before sharding!
    
    # 2. Consolidate Valid Step 1 Dates Dictionary
    shard_jsons = glob.glob('data/state/step1_dates_shard_*.json')
    step1_dates_dict = {}
    for f in shard_jsons:
        with open(f, 'r') as file:
            step1_dates_dict.update(json.load(file))
            
    # 3. Shard for Distributed Execution
    shard_tickers = all_passed_tickers[args.shard :: args.total]
    print(f"[Runner {args.shard}/{args.total}] Processing {len(shard_tickers)} tickers for Step 2.")
    
    final_buy_candidates = []
    
    for ticker in tqdm(shard_tickers, desc=f"Step 2 - Shard {args.shard}"):
        apply_anti_jitter()
        try:
            # Reconstruct the DatetimeIndex of valid dates from Step 1
            # (Removed: no longer intersecting historical dates with fundamentals)
            
            # Fetch fundamentals and run Step 2
            funds_df = fetch_fundamentals(ticker)
            if funds_df.empty:
                # If there are no fundamentals, we still need to write out the UP status from Step 1
                pred_path = os.path.join('outputs', 'predictions', f"{ticker}_prediction.json")
                if os.path.exists(pred_path):
                    with open(pred_path, 'r', encoding='utf-8') as f:
                        combined_payload = json.load(f)
                else:
                    combined_payload = {"stock_name": ticker}
                
                combined_payload["final_prediction"] = "UP"
                export_prediction_json(ticker, combined_payload)
                continue
                 
            metrics, latest_pred_class = execute_step2(funds_df)
            feature_diagnostics = metrics.pop('feature_diagnostics', {}) if metrics else {}
            final_decision = latest_pred_class == "UP"
            
            export_feature_diagnostics_json(ticker, feature_diagnostics)
            
            # Load the existing payload from Step 1
            pred_path = os.path.join('outputs', 'predictions', f"{ticker}_prediction.json")
            if os.path.exists(pred_path):
                with open(pred_path, 'r', encoding='utf-8') as f:
                    combined_payload = json.load(f)
            else:
                combined_payload = {"stock_name": ticker}
                
            combined_payload["step2_model"] = metrics
            combined_payload["final_prediction"] = "UP_FINAL_BUY" if final_decision else "UP"
            
            export_prediction_json(ticker, combined_payload)
            
            if final_decision:
                final_buy_candidates.append(ticker)
                
        except Exception as e:
            print(f"Failed {ticker} in Step 2: {e}")
            
    # Export final candidate list for this shard
    output_path = f'data/processed/final_buy_signals_shard_{args.shard}.csv'
    pd.DataFrame({'Ticker': final_buy_candidates}).to_csv(output_path, index=False)
    print(f"[Runner {args.shard}] Found {len(final_buy_candidates)} Final Buy Candidates.")

if __name__ == "__main__":
    main()
