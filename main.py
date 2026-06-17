import logging
import sys
from dotenv import load_dotenv
from src.orchestration.batch_runner import run_batch

load_dotenv()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('xetra_predictor.log')
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Xetra Two-Step Stock Prediction Engine (v4.0)")
    
    try:
        run_batch()
        logger.info("Engine run completed successfully.")
    except Exception as e:
        logger.error(f"Engine run failed with an unhandled exception: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
