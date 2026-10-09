import pandas as pd
import numpy as np
import logging
from typing import Any, Dict, Optional, Tuple
from config.settings import load_settings

logger = logging.getLogger(__name__)

RULE_LABELS = {
    1: "Rule 1 (Revenue Growth)",
    2: "Rule 2 (Profitability)",
    3: "Rule 3 (Earnings Momentum)",
    4: "Rule 4 (Cash Flow Health)",
    5: "Rule 5 (Quality of Earnings)",
    6: "Rule 6 (Free Cash Flow)",
    7: "Rule 7 (Margin Improvement)",
    8: "Rule 8 (Current Ratio)",
    9: "Rule 9 (De-leveraging)",
    10: "Rule 10 (ROE Proxy)",
}

METRIC_COLUMNS = [
    'Total Revenue', 'Net Income', 'Operating Cash Flow', 'Free Cash Flow', 'Operating Income',
    'Current Assets', 'Current Liabilities', 'Total Debt', 'Stockholders Equity'
]


def _value(row: pd.Series, key: str) -> Optional[float]:
    """Returns a statement value as float, or None if the line item is missing."""
    if key not in row.index or pd.isna(row[key]):
        return None
    return float(row[key])


def select_comparison_statement(funds_df: pd.DataFrame, tolerance_days: int) -> Tuple[pd.Series, str]:
    """
    Picks the statement the latest one is compared against.

    Comparing with the same period one year earlier removes seasonality and works for
    quarterly and half-yearly reporters alike. If no statement lies within
    365 +/- `tolerance_days` days before the latest one, the previous statement is used.

    Args:
        funds_df (pd.DataFrame): Statements sorted by date (oldest first), at least two rows.
        tolerance_days (int): Allowed deviation from exactly one year.

    Returns:
        Tuple[pd.Series, str]: The comparison statement and the basis
        ('same_period_prior_year' or 'previous_period').
    """
    latest_date = funds_df.index[-1]
    earlier = funds_df.iloc[:-1]
    distance = np.abs((earlier.index - (latest_date - pd.Timedelta(days=365))).days)
    if (distance <= tolerance_days).any():
        return earlier.iloc[int(np.argmin(distance))], 'same_period_prior_year'
    return earlier.iloc[-1], 'previous_period'


def reporting_period_days(funds_df: pd.DataFrame) -> int:
    """
    Classifies the reporting cadence from the spacing of the statements.

    Returns:
        int: 182 for half-yearly reporters, otherwise 91 (quarterly).
    """
    if len(funds_df) < 2:
        return 91
    median_gap = pd.Series(funds_df.index).diff().dt.days.median()
    return 182 if median_gap > 135 else 91


def execute_step2(funds_df: pd.DataFrame) -> Tuple[Dict[str, Any], str]:
    """
    Executes Step 2 Fundamental Ruleset Engine.

    Evaluates the latest financial statement against the same period one year earlier
    (falling back to the previous statement) using ten fundamental health rules. A rule
    whose inputs are not reported by the company (e.g. current assets for banks) is not
    applicable and is skipped. The ticker passes if at least `min_step2_applicable_rules`
    rules are applicable and the share of passed rules reaches `min_step2_score` out of 10.

    Args:
        funds_df (pd.DataFrame): The raw quarterly fundamental DataFrame.

    Returns:
        tuple: (metrics_dictionary, final_prediction_class)
    """
    logger.info("Executing Step 2 Fundamental Ruleset Engine.")
    settings = load_settings()
    min_step2_score = int(settings['min_step2_score'])
    min_applicable = int(settings['min_step2_applicable_rules'])
    tolerance_days = int(settings['step2_yoy_tolerance_days'])
    min_current_ratio = float(settings['min_current_ratio'])
    min_annual_roe = float(settings['min_annual_roe'])

    funds_df = funds_df.dropna(how='all').sort_index()
    if len(funds_df) < 2:
        logger.warning("Fewer than two financial statements available. Step 2 is not evaluable.")
        diagnostics = {"Evaluable": False, "Reason": "fewer than two financial statements"}
        return {"predicted_class": "NOT_UP", "feature_diagnostics": diagnostics}, "NOT_UP"

    latest = funds_df.iloc[-1]
    compare, comparison_basis = select_comparison_statement(funds_df, tolerance_days)
    period_days = reporting_period_days(funds_df)
    roe_threshold = min_annual_roe * period_days / 365

    rev_l, rev_c = _value(latest, 'Total Revenue'), _value(compare, 'Total Revenue')
    ni_l, ni_c = _value(latest, 'Net Income'), _value(compare, 'Net Income')
    ocf_l = _value(latest, 'Operating Cash Flow')
    fcf_l = _value(latest, 'Free Cash Flow')
    oi_l, oi_c = _value(latest, 'Operating Income'), _value(compare, 'Operating Income')
    ca_l, cl_l = _value(latest, 'Current Assets'), _value(latest, 'Current Liabilities')
    debt_l, debt_c = _value(latest, 'Total Debt'), _value(compare, 'Total Debt')
    equity_l = _value(latest, 'Stockholders Equity')

    def known(*values) -> bool:
        return all(v is not None for v in values)

    # None marks a rule whose inputs the company does not report
    rules: Dict[int, Optional[bool]] = {
        # 1. Revenue Growth vs. comparison period
        1: rev_l > rev_c if known(rev_l, rev_c) else None,
        # 2. Profitability
        2: ni_l > 0 if known(ni_l) else None,
        # 3. Earnings Momentum vs. comparison period
        3: ni_l > ni_c if known(ni_l, ni_c) else None,
        # 4. Cash Flow Health
        4: ocf_l > 0 if known(ocf_l) else None,
        # 5. Quality of Earnings: cash generation exceeds accounting profit
        5: ocf_l > ni_l if known(ocf_l, ni_l) else None,
        # 6. Free Cash Flow Generation
        6: fcf_l > 0 if known(fcf_l) else None,
        # 7. Operating Margin Improvement vs. comparison period
        7: (oi_l / rev_l > oi_c / rev_c) if known(oi_l, oi_c, rev_l, rev_c) and rev_l > 0 and rev_c > 0 else None,
        # 8. Liquidity (Current Ratio)
        8: (ca_l / cl_l > min_current_ratio) if known(ca_l, cl_l) and cl_l > 0 else None,
        # 9. De-leveraging vs. comparison period
        9: debt_l < debt_c if known(debt_l, debt_c) else None,
        # 10. ROE Proxy, pro-rated to the length of the reporting period; negative equity fails
        10: ((ni_l / equity_l > roe_threshold) if equity_l > 0 else False) if known(ni_l, equity_l) else None,
    }

    applicable = sum(1 for passed in rules.values() if passed is not None)
    rules_passed = sum(1 for passed in rules.values() if passed)
    # min_step2_score out of 10, pro-rated to the applicable rules (ceiling in integer math)
    required_score = -(-min_step2_score * applicable // 10)
    evaluable = applicable >= min_applicable
    latest_pred_class = "UP" if evaluable and rules_passed >= required_score else "NOT_UP"

    if not evaluable:
        logger.info(f"Fundamental Ruleset not evaluable: only {applicable} rules applicable (minimum {min_applicable}).")
    elif latest_pred_class == "UP":
        logger.info(f"Fundamental Ruleset Passed! Score: {rules_passed}/{applicable} (Required: {required_score})")
    else:
        logger.info(f"Fundamental Ruleset Failed. Score: {rules_passed}/{applicable} (Required: {required_score})")

    diagnostics: Dict[str, Any] = {RULE_LABELS[i]: rules[i] for i in sorted(rules)}
    diagnostics.update({
        "Total Score": int(rules_passed),
        "Applicable Rules": int(applicable),
        "Required Score": int(required_score),
        "Evaluable": bool(evaluable),
        "Comparison Basis": comparison_basis,
        "ROE Threshold": round(roe_threshold, 4),
        "Q_latest_date": str(latest.name.date()),
        "Q_prev_date": str(compare.name.date()),
        "Metrics_latest": {col: _value(latest, col) for col in METRIC_COLUMNS},
        "Metrics_prev": {col: _value(compare, col) for col in METRIC_COLUMNS},
    })

    metrics = {
        "predicted_class": latest_pred_class,
        "feature_diagnostics": diagnostics
    }

    return metrics, latest_pred_class
