import numpy as np
import pandas as pd
import pytest

from src.modeling.step2_funds import execute_step2, select_comparison_statement

HEALTHY = {
    'Total Revenue': 100.0, 'Net Income': 10.0, 'Operating Cash Flow': 15.0, 'Free Cash Flow': 8.0,
    'Operating Income': 14.0, 'Current Assets': 60.0, 'Current Liabilities': 40.0,
    'Total Debt': 50.0, 'Stockholders Equity': 200.0,
}


def _statements(dates, rows):
    return pd.DataFrame(rows, index=pd.to_datetime(dates))


def _quarterly(n=5, growth=0.05):
    dates = pd.date_range('2025-03-31', periods=n, freq='QE')
    rows = []
    for i in range(n):
        row = dict(HEALTHY)
        row['Total Revenue'] *= (1 + growth) ** i
        row['Net Income'] *= (1 + growth) ** i
        row['Operating Income'] *= (1 + growth) ** (2 * i)
        row['Total Debt'] -= i
        rows.append(row)
    return _statements(dates, rows)


def test_latest_quarter_is_compared_with_the_same_quarter_a_year_earlier():
    funds = _quarterly(n=5)
    compare, basis = select_comparison_statement(funds, tolerance_days=45)
    assert basis == 'same_period_prior_year'
    assert compare.name == funds.index[0]


def test_falls_back_to_previous_statement_without_prior_year_data():
    funds = _quarterly(n=2)
    compare, basis = select_comparison_statement(funds, tolerance_days=45)
    assert basis == 'previous_period'
    assert compare.name == funds.index[0]


def test_healthy_quarterly_reporter_passes_all_rules():
    metrics, pred = execute_step2(_quarterly(n=5))
    diag = metrics['feature_diagnostics']
    assert pred == 'UP'
    assert diag['Applicable Rules'] == 10 and diag['Required Score'] == 7 and diag['Total Score'] == 10
    assert diag['Comparison Basis'] == 'same_period_prior_year'


def test_seasonal_drop_versus_previous_quarter_is_not_penalized():
    funds = _quarterly(n=5)
    # Seasonal pattern: Q4 is strong, the latest Q1 is weaker than Q4 but better than last year's Q1
    funds.iloc[3, funds.columns.get_loc('Total Revenue')] = 500.0
    metrics, _ = execute_step2(funds)
    assert metrics['feature_diagnostics']['Rule 1 (Revenue Growth)'] is True


def test_bank_without_working_capital_or_cash_flow_items_is_scored_on_applicable_rules():
    bank_row = {k: v for k, v in HEALTHY.items()
                if k not in ('Operating Cash Flow', 'Free Cash Flow', 'Operating Income',
                             'Current Assets', 'Current Liabilities')}
    funds = _statements(pd.date_range('2025-03-31', periods=5, freq='QE'), [bank_row] * 5)
    funds['Total Revenue'] = [100, 101, 102, 103, 110.0]
    funds['Net Income'] = [10, 10, 10, 10, 12.0]
    funds['Total Debt'] = [50, 50, 50, 50, 45.0]

    metrics, pred = execute_step2(funds)
    diag = metrics['feature_diagnostics']

    assert diag['Applicable Rules'] == 5
    assert diag['Required Score'] == 4  # ceil(7 * 5 / 10)
    assert diag['Rule 8 (Current Ratio)'] is None
    assert diag['Evaluable'] is True
    assert pred == 'UP'


def test_company_with_too_few_applicable_rules_is_not_evaluable():
    funds = _statements(['2025-06-30', '2026-06-30'], [{'Net Income': 5.0, 'Total Revenue': 50.0}] * 2)
    metrics, pred = execute_step2(funds)
    assert pred == 'NOT_UP'
    assert metrics['feature_diagnostics']['Evaluable'] is False


def test_half_year_reporter_uses_prior_year_half_and_prorated_roe():
    dates = ['2024-12-31', '2025-06-30', '2025-12-31', '2026-06-30']
    rows = [dict(HEALTHY) for _ in dates]
    rows[-1]['Net Income'] = 13.0  # 13 / 200 = 6.5% per half-year > 6% (12% annualized)
    metrics, pred = execute_step2(_statements(dates, rows))
    diag = metrics['feature_diagnostics']
    assert diag['Q_prev_date'] == '2025-06-30'
    assert diag['ROE Threshold'] == pytest.approx(0.12 * 182 / 365, abs=1e-4)
    assert diag['Rule 10 (ROE Proxy)'] is True


def test_single_statement_is_not_evaluable():
    metrics, pred = execute_step2(_statements(['2026-06-30'], [HEALTHY]))
    assert pred == 'NOT_UP' and metrics['feature_diagnostics']['Evaluable'] is False
