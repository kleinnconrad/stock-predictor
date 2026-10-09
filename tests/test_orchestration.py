import json

import numpy as np
import pandas as pd
import pytest

from src.orchestration import json_exporter, steps


@pytest.fixture
def in_tmp(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_non_finite_values_are_written_as_null(in_tmp):
    json_exporter.export_prediction_json('ABC.DE', {
        'ks_cutoff': float('inf'), 'latest_prob': np.float64('nan'), 'n': np.int64(3)})
    text = (in_tmp / 'outputs' / 'predictions' / 'ABC.DE_prediction.json').read_text()
    assert json.loads(text) == {'ks_cutoff': None, 'latest_prob': None, 'n': 3}


def test_failed_serialization_keeps_the_existing_file(in_tmp):
    json_exporter.export_prediction_json('ABC.DE', {'final_prediction': 'UP'})
    with pytest.raises(TypeError):
        json_exporter.export_prediction_json('ABC.DE', {'bad': object()})
    assert json_exporter.load_prediction_json('ABC.DE') == {'final_prediction': 'UP'}


def test_step2_extends_the_step1_payload_and_diagnostics(in_tmp, monkeypatch):
    json_exporter.export_prediction_json('ABC.DE', {
        'stock_name': 'ABC.DE', 'step1_model': {'predicted_class': 'UP'}, 'final_prediction': 'UP'})
    json_exporter.export_feature_diagnostics_json('ABC.DE', {'step1_macro': {'fetched_features': ['f0']}})

    statement = {'Total Revenue': 100.0, 'Net Income': 10.0, 'Operating Cash Flow': 15.0,
                 'Free Cash Flow': 8.0, 'Operating Income': 14.0, 'Current Assets': 60.0,
                 'Current Liabilities': 40.0, 'Total Debt': 50.0, 'Stockholders Equity': 200.0}
    funds = pd.DataFrame([statement] * 5, index=pd.date_range('2025-03-31', periods=5, freq='QE'))
    funds['Total Revenue'] = [100, 101, 102, 103, 120.0]
    monkeypatch.setattr(steps, 'fetch_fundamentals', lambda ticker: funds)

    payload = steps.run_step2_for_ticker('ABC.DE')

    assert payload['step1_model'] == {'predicted_class': 'UP'}
    assert payload['final_prediction'] in ('UP', 'UP_FINAL_BUY')
    assert 'feature_diagnostics' in payload['step2_model']
    diagnostics = json_exporter.load_feature_diagnostics_json('ABC.DE')
    assert set(diagnostics) == {'step1_macro', 'step2_funds'}


def test_step2_without_fundamentals_keeps_step1_result(in_tmp, monkeypatch):
    json_exporter.export_prediction_json('ABC.DE', {'stock_name': 'ABC.DE', 'final_prediction': 'UP'})
    monkeypatch.setattr(steps, 'fetch_fundamentals', lambda ticker: pd.DataFrame())
    assert steps.run_step2_for_ticker('ABC.DE')['final_prediction'] == 'UP'
