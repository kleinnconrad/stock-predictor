import os
import sys

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
import glob
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from src.orchestration.logging_setup import configure_logging
from src.orchestration.steps import run_step2_for_ticker

logger = logging.getLogger(__name__)

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
    configure_logging()

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

    # 2. Shard for Distributed Execution
    shard_tickers = all_passed_tickers[args.shard :: args.total]
    logger.info(f"[Runner {args.shard}/{args.total}] Processing {len(shard_tickers)} tickers for Step 2.")

    final_buy_candidates = []

    for ticker in tqdm(shard_tickers, desc=f"Step 2 - Shard {args.shard}"):
        apply_anti_jitter()
        try:
            payload = run_step2_for_ticker(ticker)
            if payload['final_prediction'] == 'UP_FINAL_BUY':
                final_buy_candidates.append(ticker)
        except Exception:
            logger.exception(f"Failed {ticker} in Step 2")

    # The final buy list is built from the consolidated payloads by scripts/04_consolidate.py
    logger.info(f"[Runner {args.shard}] Found {len(final_buy_candidates)} Final Buy Candidates.")

if __name__ == "__main__":
    main()
