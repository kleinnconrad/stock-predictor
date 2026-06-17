import json
import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

def export_prediction_json(ticker: str, payload_data: Dict[str, Any]):
    """
    Exports the final prediction and model parameters to a strict JSON schema.
    
    Args:
        ticker (str): The stock ticker.
        payload_data (Dict[str, Any]): The formatted payload data.
    """
    output_dir = os.path.join('outputs', 'predictions')
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, f"{ticker}_prediction.json")
    
    try:
        # payload_data should already be cast to native Python floats inside the step functions,
        # but just in case, json.dumps will fail if not, which is a good strict check.
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dumps(payload_data) # Dry run test
            json.dump(payload_data, f, indent=2)
            
        logger.info(f"Successfully exported JSON prediction for {ticker} to {file_path}")
    except TypeError as e:
        logger.error(f"Failed to serialize JSON for {ticker} - likely a numpy type wasn't cast: {e}")
        raise
    except Exception as e:
        logger.error(f"Error writing JSON for {ticker}: {e}")
        raise

def export_feature_diagnostics_json(ticker: str, diagnostics_data: Dict[str, Any]):
    """
    Exports the feature selection diagnostics to a JSON file.
    
    Args:
        ticker (str): The stock ticker.
        diagnostics_data (Dict[str, Any]): The formatted diagnostics data.
    """
    output_dir = os.path.join('outputs', 'diagnostics')
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, f"{ticker}_feature_diagnostics.json")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(diagnostics_data, f, indent=2)
            
        logger.info(f"Successfully exported feature diagnostics for {ticker} to {file_path}")
    except Exception as e:
        logger.error(f"Error writing feature diagnostics JSON for {ticker}: {e}")
        raise
