"""
Creates random dummy reports for previewing the dashboard's uplift charts.

The reports are written to a separate directory (data/processed/history_dummy by
default) so they can never be mistaken for real predictions. Evaluate them with:

    uv run python -m src.processing.uplift_evaluator --history-dir data/processed/history_dummy --out <file>
"""
import os
import json
import random
import argparse
from datetime import datetime, timedelta, timezone

def create_dummy_data(history_dir: str):
    os.makedirs(history_dir, exist_ok=True)

    today = datetime.now(timezone.utc).date()
    intervals = [30, 90, 180]

    # We need real tickers so yfinance can fetch them.
    tickers = ["SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "BMW.DE", "MBG.DE", "BAS.DE", "BAYN.DE", "VOW3.DE", "MUV2.DE"]

    for days in intervals:
        target_date = today - timedelta(days=days)
        predictions = []
        for ticker in tickers:
            pred = random.choice(["UP_FINAL_BUY", "UP", "NOT_UP", "NOT_UP"]) # Skew a bit towards NOT_UP
            predictions.append({
                "stock_name": ticker,
                "final_prediction": pred
            })

        report = {
            "execution_date": target_date.isoformat() + "T00:00:00Z",
            "parameters": {"dummy": True},
            "predictions": predictions
        }

        month_str = target_date.strftime("%Y-%m")
        month_dir = os.path.join(history_dir, month_str)
        os.makedirs(month_dir, exist_ok=True)

        file_path = os.path.join(month_dir, f"report_{target_date.isoformat()}.json")
        with open(file_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Created dummy report for {days} days ago at {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history-dir", default=os.path.join("data", "processed", "history_dummy"),
                        help="Target directory (never the real data/processed/history)")
    args = parser.parse_args()
    if os.path.normpath(args.history_dir) == os.path.normpath(os.path.join("data", "processed", "history")):
        parser.error("Refusing to write dummy reports into the real history directory.")
    create_dummy_data(args.history_dir)
