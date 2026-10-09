import json
from datetime import date

import pandas as pd

from src.processing import uplift_evaluator


def test_empty_cohorts_are_missing_not_zero():
    predictions = [{'stock_name': 'AAA', 'final_prediction': 'NOT_UP'},
                   {'stock_name': 'BBB', 'final_prediction': 'UP'}]
    metrics, counts = uplift_evaluator.cohort_returns(predictions, {'AAA': 0.1, 'BBB': -0.1})
    assert metrics['UP_FINAL_BUY'] is None and counts['UP_FINAL_BUY'] == 0
    assert metrics['baseline'] == 0.0 and counts['baseline'] == 2


def test_returns_start_at_the_matched_reports_own_date(tmp_path, monkeypatch):
    # The 1-month lookback (2026-09-09) matches a report dated 2026-09-01 (within 15 days)
    report_dir = tmp_path / 'history' / '2026-09'
    report_dir.mkdir(parents=True)
    (report_dir / 'report_2026-09-01.json').write_text(json.dumps({
        'execution_date': '2026-09-01T01:00:00Z',
        'predictions': [{'stock_name': 'AAA', 'final_prediction': 'UP_FINAL_BUY'}],
    }))
    index = pd.bdate_range('2026-08-27', '2026-10-08')
    prices = pd.DataFrame({'AAA': 100.0}, index=index)
    prices.loc[prices.index >= '2026-09-01', 'AAA'] = 110.0  # level on the report date
    prices.iloc[-1, 0] = 121.0
    monkeypatch.setattr(uplift_evaluator, 'download_closes', lambda tickers, start, end: prices)

    out = uplift_evaluator.evaluate_uplift(str(tmp_path / 'history'), str(tmp_path / 'uplift.json'),
                                           today=date(2026, 10, 9))

    assert out['1m']['report_date'] == '2026-09-01'
    assert abs(out['1m']['metrics']['UP_FINAL_BUY'] - 0.10) < 1e-9  # 110 -> 121, not 100 -> 121
    assert '3m' not in out
