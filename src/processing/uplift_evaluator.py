import os
import glob
import json
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_closest_report(history_dir, target_date, tolerance_days=10):
    files = glob.glob(os.path.join(history_dir, "report_*.json"))
    closest_file = None
    min_diff = timedelta(days=9999)
    for f in files:
        basename = os.path.basename(f)
        date_str = basename.replace("report_", "").replace(".json", "")
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            diff = abs(file_date - target_date)
            if diff <= timedelta(days=tolerance_days) and diff < min_diff:
                min_diff = diff
                closest_file = f
        except ValueError:
            continue
    return closest_file

def evaluate_uplift():
    history_dir = os.path.join("data", "processed", "history")
    if not os.path.exists(history_dir):
        logger.warning(f"History directory {history_dir} does not exist.")
        return
        
    today = datetime.utcnow().date()
    intervals = {
        "1m": 30,
        "3m": 90,
        "6m": 180
    }
    
    report_data = {}
    
    for label, days in intervals.items():
        target_date = today - timedelta(days=days)
        report_file = find_closest_report(history_dir, target_date)
        if not report_file:
            logger.info(f"No report found close to {days} days ago ({target_date}).")
            continue
            
        with open(report_file, 'r') as f:
            data = json.load(f)
            
        predictions = data.get("predictions", [])
        if not predictions:
            continue
            
        tickers = [p["stock_name"] for p in predictions if p.get("stock_name")]
        if not tickers:
            continue
            
        start_date = target_date - timedelta(days=5)
        end_date = today + timedelta(days=2)
        
        try:
            dl = yf.download(tickers, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), progress=False)
            if dl.empty:
                continue
            
            if 'Close' in dl:
                prices = dl['Close']
            else:
                prices = dl
                
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=tickers[0])
        except Exception as e:
            logger.error(f"Failed to download prices: {e}")
            continue
            
        returns = {}
        for ticker in tickers:
            try:
                # yfinance returns MultiIndex columns if multiple tickers.
                # If 'Close' was a top level, we might have columns as tickers now.
                if ticker in prices.columns:
                    ticker_prices = prices[ticker].dropna()
                else:
                    continue
                    
                if len(ticker_prices) >= 2:
                    # Find closest to target date
                    hist_idx = ticker_prices.index.get_indexer([pd.Timestamp(target_date)], method='nearest')[0]
                    hist_price = ticker_prices.iloc[hist_idx]
                    curr_price = ticker_prices.iloc[-1]
                    
                    # Convert to scalar float
                    hist_price = float(hist_price.iloc[0]) if isinstance(hist_price, pd.Series) else float(hist_price)
                    curr_price = float(curr_price.iloc[0]) if isinstance(curr_price, pd.Series) else float(curr_price)
                    
                    if hist_price > 0:
                        returns[ticker] = (curr_price - hist_price) / hist_price
            except Exception as e:
                continue
                
        if not returns:
            continue
            
        cohorts = {"baseline": [], "UP_FINAL_BUY": [], "UP": [], "NOT_UP": []}
        for p in predictions:
            ticker = p.get("stock_name")
            pred = p.get("final_prediction", "NOT_UP")
            if ticker in returns:
                ret = returns[ticker]
                cohorts["baseline"].append(ret)
                if pred in cohorts:
                    cohorts[pred].append(ret)
                else:
                    cohorts["NOT_UP"].append(ret)
                    
        metrics = {}
        for cohort, vals in cohorts.items():
            if vals:
                metrics[cohort] = sum(vals) / len(vals)
            else:
                metrics[cohort] = 0.0
                
        report_data[label] = {
            "metrics": metrics,
            "date": data.get("execution_date", target_date.isoformat()),
            "is_dummy": data.get("parameters", {}).get("dummy", False)
        }
        
    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "uplift_report.json")
    with open(out_file, 'w') as f:
        json.dump(report_data, f, indent=2)
        
    logger.info(f"Uplift report generated at {out_file}")

if __name__ == "__main__":
    evaluate_uplift()
