import json
import os

with open("data/processed/full_batch_report.json", "r") as f:
    report = json.load(f)

for pred in report["predictions"]:
    ticker = pred["stock_name"]
    diag_path = f"outputs/diagnostics/{ticker}_feature_diagnostics.json"
    if os.path.exists(diag_path):
        with open(diag_path, "r") as df:
            diag = json.load(df)
            if "step2_funds" in diag:
                if "step2_model" not in pred:
                    pred["step2_model"] = {}
                pred["step2_model"]["feature_diagnostics"] = diag["step2_funds"]
            elif "Rule 1 (Revenue Growth)" in diag:
                if "step2_model" not in pred:
                    pred["step2_model"] = {}
                pred["step2_model"]["feature_diagnostics"] = diag

with open("data/processed/full_batch_report.json", "w") as f:
    json.dump(report, f, indent=2)
print("Patched full_batch_report.json")
