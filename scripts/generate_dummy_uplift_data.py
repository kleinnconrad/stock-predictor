import os
import json
import random
from datetime import datetime, timedelta

def create_dummy_data():
    history_dir = os.path.join("data", "processed", "history")
    os.makedirs(history_dir, exist_ok=True)
    
    today = datetime.utcnow().date()
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
        
        file_path = os.path.join(history_dir, f"report_{target_date.isoformat()}.json")
        with open(file_path, "w") as f:
            json.dump(report, f, indent=2)
            
        print(f"Created dummy report for {days} days ago at {file_path}")

if __name__ == "__main__":
    create_dummy_data()
