import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def execute_step2(funds_df: pd.DataFrame) -> Tuple[Dict[str, Any], str]:
    """
    Executes Step 2 Fundamental Ruleset Engine.
    
    Evaluates the two most recent quarterly financial statements against a strict 
    fundamental health checklist.
    
    Args:
        funds_df (pd.DataFrame): The raw quarterly fundamental DataFrame.
        
    Returns:
        tuple: (metrics_dictionary, final_prediction_class)
    """
    logger.info("Executing Step 2 Fundamental Ruleset Engine.")
    
    if funds_df.empty or len(funds_df) < 2:
        logger.warning("Not enough quarterly fundamental data to evaluate ruleset.")
        return {}, "NOT_UP"
        
    # Get the two most recent quarters
    q_latest = funds_df.iloc[-1]
    q_prev = funds_df.iloc[-2]
    
    def safe_get(series, key):
        val = series.get(key, np.nan)
        return float(val) if pd.notna(val) else 0.0
    
    # 1. Revenue Growth: Total Revenue (Q_latest) > Total Revenue (Q_prev)
    rev_latest = safe_get(q_latest, 'Total Revenue')
    rev_prev = safe_get(q_prev, 'Total Revenue')
    rule_1_pass = rev_latest > rev_prev
    
    # 2. Profitability: Net Income (Q_latest) > 0
    ni_latest = safe_get(q_latest, 'Net Income')
    rule_2_pass = ni_latest > 0
    
    # 3. Earnings Momentum: Net Income (Q_latest) > Net Income (Q_prev)
    ni_prev = safe_get(q_prev, 'Net Income')
    rule_3_pass = ni_latest > ni_prev
    
    # 4. Cash Flow Health: Operating Cash Flow (Q_latest) > 0
    ocf_latest = safe_get(q_latest, 'Operating Cash Flow')
    rule_4_pass = ocf_latest > 0
    
    rules_passed = sum([rule_1_pass, rule_2_pass, rule_3_pass, rule_4_pass])
    
    # Require at least 3 out of 4 rules to pass
    latest_pred_class = "UP" if rules_passed >= 3 else "NOT_UP"
    
    if latest_pred_class == "UP":
        logger.info(f"Fundamental Ruleset Passed! Score: {rules_passed}/4")
    else:
        logger.info(f"Fundamental Ruleset Failed. Score: {rules_passed}/4")
    
    diagnostics = {
        "Rule 1 (Revenue Growth)": bool(rule_1_pass),
        "Rule 2 (Profitability)": bool(rule_2_pass),
        "Rule 3 (Earnings Momentum)": bool(rule_3_pass),
        "Rule 4 (Cash Flow Health)": bool(rule_4_pass),
        "Total Score": int(rules_passed),
        "Required Score": 3,
        "Q_latest_date": str(q_latest.name.date()),
        "Q_prev_date": str(q_prev.name.date()),
        "Metrics_latest": {
            "Total Revenue": rev_latest,
            "Net Income": ni_latest,
            "Operating Cash Flow": ocf_latest
        },
        "Metrics_prev": {
            "Total Revenue": rev_prev,
            "Net Income": ni_prev
        }
    }
    
    metrics = {
        "predicted_class": latest_pred_class,
        "feature_diagnostics": diagnostics
    }
    
    return metrics, latest_pred_class
