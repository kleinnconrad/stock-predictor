"""
Consolidation of per-ticker prediction payloads into the batch report, the buy-signal
list and the dated history archive. Used by scripts/04_consolidate.py (GitHub Actions)
and by the local batch runner.
"""
import glob
import json
import logging
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .json_exporter import write_json_atomic

logger = logging.getLogger(__name__)

VALID_PREDICTIONS = ("UP", "NOT_UP", "UP_FINAL_BUY")
REPORT_FILE_PATTERN = re.compile(r"report_(\d{4}-\d{2}-\d{2})\.json$")


def _preference(payload: Dict[str, Any]) -> tuple:
    # A payload that went through Step 2 is more complete; a buy beats a Step 2 rejection
    return ('step2_model' in payload, payload.get('final_prediction') == 'UP_FINAL_BUY')


def consolidate_predictions(prediction_files: Iterable[str]) -> List[Dict[str, Any]]:
    """
    Merges prediction payloads from all shards into one payload per ticker.

    Every Step 2 shard artifact contains the Step 1 payloads of all tickers but the
    Step 2 results of its own shard only, so the most complete payload is kept.

    Args:
        prediction_files (Iterable[str]): Paths of *_prediction.json files.

    Returns:
        List[Dict[str, Any]]: One payload per ticker, sorted by ticker.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for path in sorted(prediction_files):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to read {path}: {e}")
            continue
        ticker = payload.get('stock_name')
        if not ticker:
            continue
        if payload.get('final_prediction') not in VALID_PREDICTIONS:
            payload['final_prediction'] = 'NOT_UP'
        current = merged.get(ticker)
        if current is None or _preference(payload) > _preference(current):
            merged[ticker] = payload
    return [merged[ticker] for ticker in sorted(merged)]


def build_report(predictions: List[Dict[str, Any]], settings: Dict[str, Any],
                 now: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Builds the batch report published to the dashboard.

    Args:
        predictions (List[Dict[str, Any]]): Consolidated payloads.
        settings (Dict[str, Any]): Settings used for the run.
        now (Optional[datetime]): Execution time (UTC); defaults to the current time.

    Returns:
        Dict[str, Any]: The report.
    """
    now = now or datetime.now(timezone.utc)
    return {
        "execution_date": now.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
        "parameters": settings,
        "predictions": predictions,
    }


def write_final_buy_signals(predictions: List[Dict[str, Any]], path: str) -> List[str]:
    """
    Writes the tickers that passed both steps to a CSV with a `Ticker` column.

    Returns:
        List[str]: The buy candidates.
    """
    buys = sorted(p['stock_name'] for p in predictions if p.get('final_prediction') == 'UP_FINAL_BUY')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    pd.DataFrame({'Ticker': buys}).to_csv(path, index=False)
    return buys


def archive_report(report: Dict[str, Any], history_dir: str, report_date: date) -> str:
    """
    Stores the report as history/YYYY-MM/report_YYYY-MM-DD.json.

    Returns:
        str: Path of the archived report.
    """
    path = os.path.join(history_dir, report_date.strftime('%Y-%m'), f"report_{report_date.isoformat()}.json")
    write_json_atomic(path, report)
    return path


def report_date_from_path(path: str) -> Optional[date]:
    """Returns the date encoded in a report_YYYY-MM-DD.json file name, or None."""
    match = REPORT_FILE_PATTERN.search(os.path.basename(path))
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def prune_history(history_dir: str, today: date, retention_days: int) -> List[str]:
    """
    Deletes archived reports older than the retention window.

    The age is taken from the date in the file name. File modification times are not
    usable: every file of a fresh git checkout has the checkout time.

    Args:
        history_dir (str): The history directory.
        today (date): Reference date.
        retention_days (int): Reports dated before today - retention_days are removed.

    Returns:
        List[str]: Removed files.
    """
    cutoff = today - timedelta(days=retention_days)
    removed = []
    for path in glob.glob(os.path.join(history_dir, '*', 'report_*.json')):
        report_date = report_date_from_path(path)
        if report_date is not None and report_date < cutoff:
            os.remove(path)
            removed.append(path)
    for month_dir in glob.glob(os.path.join(history_dir, '*')):
        if os.path.isdir(month_dir) and not os.listdir(month_dir):
            os.rmdir(month_dir)
    if removed:
        logger.info(f"Pruned {len(removed)} archived reports older than {cutoff}.")
    return removed


def publish_report(predictions: List[Dict[str, Any]], settings: Dict[str, Any], out_dir: str,
                   now: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Writes full_batch_report.json and final_buy_signals.csv to `out_dir`, archives the
    report under `out_dir`/history and prunes reports older than `history_retention_days`.

    Args:
        predictions (List[Dict[str, Any]]): Consolidated payloads.
        settings (Dict[str, Any]): Settings used for the run.
        out_dir (str): Output directory (data/processed in production).
        now (Optional[datetime]): Execution time (UTC); defaults to the current time.

    Returns:
        Dict[str, Any]: The published report.
    """
    now = now or datetime.now(timezone.utc)
    report = build_report(predictions, settings, now)
    write_json_atomic(os.path.join(out_dir, 'full_batch_report.json'), report)
    buys = write_final_buy_signals(predictions, os.path.join(out_dir, 'final_buy_signals.csv'))
    history_dir = os.path.join(out_dir, 'history')
    archive_report(report, history_dir, now.date())
    prune_history(history_dir, now.date(), int(settings['history_retention_days']))
    logger.info(f"Published report with {len(predictions)} predictions and {len(buys)} buy candidates to {out_dir}.")
    return report
