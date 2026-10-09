"""
Backtests archived batch reports: for the reports closest to 1, 3 and 6 months ago,
compares the realized return of each prediction cohort with the equal-weighted return
of all evaluated stocks (baseline).

Run from the repository root: `uv run python -m src.processing.uplift_evaluator`.
"""
import argparse
import glob
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from config.settings import load_settings
from src.orchestration.json_exporter import write_json_atomic
from src.orchestration.report import report_date_from_path

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_DIR = os.path.join("data", "processed", "history")
DEFAULT_OUT_FILE = os.path.join("data", "processed", "uplift_report.json")
INTERVALS = {"1m": 30, "3m": 90, "6m": 180}
COHORTS = ("baseline", "UP_FINAL_BUY", "UP", "NOT_UP")


def find_closest_report(history_dir: str, target_date: date, tolerance_days: int) -> Optional[Tuple[str, date]]:
    """
    Finds the archived report dated closest to `target_date` within the tolerance.

    Returns:
        Optional[Tuple[str, date]]: (path, report date) or None.
    """
    best = None
    for path in glob.glob(os.path.join(history_dir, "*", "report_*.json")):
        report_date = report_date_from_path(path)
        if report_date is None:
            continue
        distance = abs((report_date - target_date).days)
        if distance <= tolerance_days and (best is None or distance < best[0]):
            best = (distance, path, report_date)
    return (best[1], best[2]) if best else None


def price_on_or_after(prices: pd.Series, day: date) -> Optional[float]:
    """Returns the first close on or after `day` (the first price investable after the report)."""
    prices = prices.dropna()
    prices = prices[prices.index >= pd.Timestamp(day)]
    return float(prices.iloc[0]) if not prices.empty else None


def download_closes(tickers: List[str], start: date, end: date) -> pd.DataFrame:
    """Downloads daily closes (one column per ticker) for [start, end)."""
    data = yf.download(tickers, start=start.isoformat(), end=end.isoformat(), progress=False)
    if data.empty:
        return pd.DataFrame()
    closes = data['Close'] if 'Close' in data else data
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(name=tickers[0])
    return closes


def cohort_returns(predictions: List[dict], returns: Dict[str, float]) -> Tuple[Dict[str, Optional[float]], Dict[str, int]]:
    """
    Averages the realized returns per prediction cohort.

    Returns:
        Tuple[Dict[str, Optional[float]], Dict[str, int]]: Mean return per cohort (None for
        an empty cohort, which is not the same as a 0% return) and the cohort sizes.
    """
    groups: Dict[str, List[float]] = {cohort: [] for cohort in COHORTS}
    for prediction in predictions:
        ticker = prediction.get("stock_name")
        if ticker not in returns:
            continue
        groups["baseline"].append(returns[ticker])
        cohort = prediction.get("final_prediction", "NOT_UP")
        groups[cohort if cohort in groups else "NOT_UP"].append(returns[ticker])
    metrics = {cohort: (sum(vals) / len(vals) if vals else None) for cohort, vals in groups.items()}
    counts = {cohort: len(vals) for cohort, vals in groups.items()}
    return metrics, counts


def evaluate_uplift(history_dir: str = DEFAULT_HISTORY_DIR, out_file: str = DEFAULT_OUT_FILE,
                    today: Optional[date] = None) -> Dict[str, dict]:
    """
    Evaluates the archived reports closest to 1, 3 and 6 months ago and writes the uplift report.

    Returns are measured from the first close on or after each report's own date to the
    latest close before today.

    Args:
        history_dir (str): Directory with archived reports (history/YYYY-MM/report_*.json).
        out_file (str): Output JSON path.
        today (Optional[date]): Reference date (UTC today if omitted).

    Returns:
        Dict[str, dict]: The uplift report keyed by horizon ('1m', '3m', '6m').
    """
    if not os.path.exists(history_dir):
        logger.warning(f"History directory {history_dir} does not exist.")
        return {}

    today = today or datetime.now(timezone.utc).date()
    tolerance_days = int(load_settings()['uplift_match_tolerance_days'])
    report_data = {}

    for label, days in INTERVALS.items():
        match = find_closest_report(history_dir, today - timedelta(days=days), tolerance_days)
        if not match:
            logger.info(f"No report found within {tolerance_days} days of {days} days ago.")
            continue
        path, report_date = match
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        predictions = data.get("predictions", [])
        tickers = sorted({p["stock_name"] for p in predictions if p.get("stock_name")})
        if not tickers:
            continue

        try:
            closes = download_closes(tickers, report_date - timedelta(days=5), today)
        except Exception as e:
            logger.error(f"Failed to download prices for the {label} report: {e}")
            continue

        returns = {}
        for ticker in tickers:
            if ticker not in closes.columns:
                continue
            series = closes[ticker].dropna()
            start_price = price_on_or_after(series, report_date)
            if start_price is None or start_price <= 0:
                continue
            returns[ticker] = float(series.iloc[-1]) / start_price - 1.0

        if not returns:
            continue

        metrics, counts = cohort_returns(predictions, returns)
        report_data[label] = {
            "metrics": metrics,
            "counts": counts,
            "date": data.get("execution_date", report_date.isoformat()),
            "report_date": report_date.isoformat(),
            "is_dummy": data.get("parameters", {}).get("dummy", False),
        }

    write_json_atomic(out_file, report_data)
    logger.info(f"Uplift report generated at {out_file}")
    return report_data


if __name__ == "__main__":
    from src.orchestration.logging_setup import configure_logging

    parser = argparse.ArgumentParser(description="Evaluates archived predictions against realized returns.")
    parser.add_argument("--history-dir", default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT_FILE)
    args = parser.parse_args()
    configure_logging()
    evaluate_uplift(history_dir=args.history_dir, out_file=args.out)
