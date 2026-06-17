import logging
import sys
import os
import argparse
from dotenv import load_dotenv
from src.orchestration.batch_runner import run_batch, run_single

load_dotenv()

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join('logs', 'xetra_predictor.log'))
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Xetra Two-Step Stock Prediction Engine")
    parser.add_argument("--ticker", type=str, help="Run the model for a single specific ticker (e.g., SAP.DE)", default=None)
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Xetra Two-Step Stock Prediction Engine (v5.0)")
    
    try:
        if args.ticker:
            logger.info(f"Single Ticker Mode activated for: {args.ticker}")
            run_single(args.ticker)
        else:
            logger.info("Batch Mode activated.")
            run_batch()
            
        logger.info("Engine run completed successfully.")
    except Exception as e:
        logger.error(f"Engine run failed with an unhandled exception: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
