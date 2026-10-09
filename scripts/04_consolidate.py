import os
import sys

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import logging
import argparse
from config.settings import load_settings
from src.orchestration.logging_setup import configure_logging
from src.orchestration.report import consolidate_predictions, publish_report

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Consolidates the shard outputs into the published batch report.")
    parser.add_argument("--predictions-glob", default="step2_artifacts/*/outputs/predictions/*.json",
                        help="Glob of the prediction payloads to consolidate")
    parser.add_argument("--out-dir", default=os.path.join("data", "processed"),
                        help="Directory for full_batch_report.json, final_buy_signals.csv and history/")
    args = parser.parse_args()
    configure_logging()

    prediction_files = glob.glob(args.predictions_glob)
    if not prediction_files:
        raise FileNotFoundError(f"No prediction payloads match {args.predictions_glob}")

    predictions = consolidate_predictions(prediction_files)
    publish_report(predictions, load_settings(), args.out_dir)

if __name__ == "__main__":
    main()
