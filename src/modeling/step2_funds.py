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
    
    # 5. Quality of Earnings: Operating Cash Flow (Q_latest) > Net Income (Q_latest)
    rule_5_pass = ocf_latest > ni_latest
    
    # 6. Free Cash Flow Generation: FCF (Q_latest) > 0
    fcf_latest = safe_get(q_latest, 'Free Cash Flow')
    rule_6_pass = fcf_latest > 0
    
    # 7. Operating Margin Improvement: (OpInc / Rev)_latest > (OpInc / Rev)_prev
    op_inc_latest = safe_get(q_latest, 'Operating Income')
    op_inc_prev = safe_get(q_prev, 'Operating Income')
    margin_latest = op_inc_latest / rev_latest if rev_latest > 0 else -1
    margin_prev = op_inc_prev / rev_prev if rev_prev > 0 else -1
    rule_7_pass = margin_latest > margin_prev
    
    # 8. Liquidity (Current Ratio): Current Assets / Current Liab > 1.2
    ca_latest = safe_get(q_latest, 'Current Assets')
    cl_latest = safe_get(q_latest, 'Current Liabilities')
    current_ratio = ca_latest / cl_latest if cl_latest > 0 else 0
    rule_8_pass = current_ratio > 1.2
    
    # 9. De-leveraging: Total Debt (Q_latest) < Total Debt (Q_prev)
    debt_latest = safe_get(q_latest, 'Total Debt')
    debt_prev = safe_get(q_prev, 'Total Debt')
    rule_9_pass = debt_latest < debt_prev
    
    # 10. ROE Proxy: Net Income / Stockholders Equity > 0.03
    equity_latest = safe_get(q_latest, 'Stockholders Equity')
    roe_latest = ni_latest / equity_latest if equity_latest > 0 else -1
    rule_10_pass = roe_latest > 0.03
    
    rules_passed = sum([
        rule_1_pass, rule_2_pass, rule_3_pass, rule_4_pass,
        rule_5_pass, rule_6_pass, rule_7_pass, rule_8_pass,
        rule_9_pass, rule_10_pass
    ])
    
    # Require at least 7 out of 10 rules to pass
    latest_pred_class = "UP" if rules_passed >= 7 else "NOT_UP"
    
    if latest_pred_class == "UP":
        logger.info(f"Fundamental Ruleset Passed! Score: {rules_passed}/10")
    else:
        logger.info(f"Fundamental Ruleset Failed. Score: {rules_passed}/10")
    
    diagnostics = {
        "Rule 1 (Revenue Growth)": bool(rule_1_pass),
        "Rule 2 (Profitability)": bool(rule_2_pass),
        "Rule 3 (Earnings Momentum)": bool(rule_3_pass),
        "Rule 4 (Cash Flow Health)": bool(rule_4_pass),
        "Rule 5 (Quality of Earnings)": bool(rule_5_pass),
        "Rule 6 (Free Cash Flow)": bool(rule_6_pass),
        "Rule 7 (Margin Improvement)": bool(rule_7_pass),
        "Rule 8 (Current Ratio)": bool(rule_8_pass),
        "Rule 9 (De-leveraging)": bool(rule_9_pass),
        "Rule 10 (ROE Proxy)": bool(rule_10_pass),
        "Total Score": int(rules_passed),
        "Required Score": 7,
        "Q_latest_date": str(q_latest.name.date()),
        "Q_prev_date": str(q_prev.name.date()),
        "Metrics_latest": {
            "Total Revenue": rev_latest,
            "Net Income": ni_latest,
            "Operating Cash Flow": ocf_latest,
            "Free Cash Flow": fcf_latest,
            "Operating Income": op_inc_latest,
            "Current Assets": ca_latest,
            "Current Liabilities": cl_latest,
            "Total Debt": debt_latest,
            "Stockholders Equity": equity_latest
        },
        "Metrics_prev": {
            "Total Revenue": rev_prev,
            "Net Income": ni_prev,
            "Operating Income": op_inc_prev,
            "Total Debt": debt_prev
        }
    }
    
    metrics = {
        "predicted_class": latest_pred_class,
        "feature_diagnostics": diagnostics
    }
    
    return metrics, latest_pred_class
