import os
import sys

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from config.settings import load_settings
from src.ingestion.global_macro import fetch_global_macro_universe
from src.orchestration.logging_setup import configure_logging
from src.orchestration.steps import history_window, run_step1_for_ticker

logger = logging.getLogger(__name__)

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
    configure_logging()

    os.makedirs('data/state', exist_ok=True)

    if not os.path.exists('data/state/qualified_tickers.csv'):
        raise FileNotFoundError("qualified_tickers.csv missing! Did Action 1 run and commit the ticker list?")

    tickers_df = pd.read_csv('data/state/qualified_tickers.csv')
    if 'Company' not in tickers_df.columns:
        tickers_df['Company'] = "Unknown Company"

    all_tickers = tickers_df.to_dict('records')
    all_tickers.sort(key=lambda x: x['Ticker']) # Ensure deterministic ordering

    # Mathematical Sharding for Distributed GitHub Actions
    shard_tickers = all_tickers[args.shard :: args.total]
    logger.info(f"[Runner {args.shard}/{args.total}] Processing {len(shard_tickers)} tickers.")

    settings = load_settings()
    _, fetch_years = history_window(settings)
    logger.info("Pre-caching Global Macro Universe...")
    macro_df = fetch_global_macro_universe(history_years=fetch_years)

    passed_tickers = []
    for row in tqdm(shard_tickers, desc=f"Step 1 - Shard {args.shard}"):
        ticker = row['Ticker']
        apply_anti_jitter()
        try:
            payload = run_step1_for_ticker(ticker, row['Company'], macro_df, settings)
            if payload and payload['final_prediction'] == 'UP':
                passed_tickers.append(ticker)
        except Exception:
            logger.exception(f"Failed {ticker} in Step 1")

    # Save State Artifacts for Step 2
    output_path = f'data/state/step1_passed_shard_{args.shard}.csv'
    pd.DataFrame({'Ticker': passed_tickers}).to_csv(output_path, index=False)
    logger.info(f"[Runner {args.shard}] Saved {len(passed_tickers)} passing tickers.")

if __name__ == "__main__":
    main()
