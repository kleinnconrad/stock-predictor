import json
from datetime import date, datetime, timezone

import pandas as pd

from src.orchestration.report import (consolidate_predictions, prune_history, publish_report,
                                      report_date_from_path)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return str(path)


def test_consolidation_keeps_the_most_complete_payload_per_ticker(tmp_path):
    files = [
        # Shard 0 artifact: Step 2 ran for AAA, BBB only has its Step 1 payload
        _write(tmp_path / 's0' / 'AAA.json', {'stock_name': 'AAA', 'final_prediction': 'UP_FINAL_BUY', 'step2_model': {}}),
        _write(tmp_path / 's0' / 'BBB.json', {'stock_name': 'BBB', 'final_prediction': 'UP'}),
        # Shard 1 artifact: Step 2 ran for BBB
        _write(tmp_path / 's1' / 'AAA.json', {'stock_name': 'AAA', 'final_prediction': 'UP'}),
        _write(tmp_path / 's1' / 'BBB.json', {'stock_name': 'BBB', 'final_prediction': 'UP', 'step2_model': {'x': 1}}),
        _write(tmp_path / 's1' / 'CCC.json', {'stock_name': 'CCC', 'final_prediction': 'PENDING'}),
    ]
    merged = {p['stock_name']: p for p in consolidate_predictions(files)}
    assert merged['AAA']['final_prediction'] == 'UP_FINAL_BUY'
    assert merged['BBB']['step2_model'] == {'x': 1}
    assert merged['CCC']['final_prediction'] == 'NOT_UP'


def test_history_is_pruned_by_report_date_not_file_age(tmp_path):
    old = _write(tmp_path / '2026-01' / 'report_2026-01-05.json', {})
    kept = _write(tmp_path / '2026-06' / 'report_2026-06-01.json', {})
    removed = prune_history(str(tmp_path), today=date(2026, 10, 9), retention_days=200)
    assert removed == [old]
    assert not (tmp_path / '2026-01').exists()
    assert (tmp_path / '2026-06' / 'report_2026-06-01.json').exists()
    assert report_date_from_path(kept) == date(2026, 6, 1)


def test_publish_writes_report_buy_list_and_archive(tmp_path):
    predictions = [{'stock_name': 'AAA', 'final_prediction': 'UP_FINAL_BUY'},
                   {'stock_name': 'BBB', 'final_prediction': 'NOT_UP'}]
    now = datetime(2026, 10, 9, 1, 30, tzinfo=timezone.utc)
    publish_report(predictions, {'history_retention_days': 200}, str(tmp_path), now=now)

    report = json.loads((tmp_path / 'full_batch_report.json').read_text())
    assert report['execution_date'] == '2026-10-09T01:30:00.000000Z'
    assert pd.read_csv(tmp_path / 'final_buy_signals.csv')['Ticker'].tolist() == ['AAA']
    assert (tmp_path / 'history' / '2026-10' / 'report_2026-10-09.json').exists()


def test_profile_cache_keeps_generated_descriptions_only(tmp_path):
    from src.orchestration.report import update_profile_cache

    cache = tmp_path / 'company_profiles_cache.json'
    cache.write_text(json.dumps({'OLD': {'full_name': 'Old AG', 'description': 'Kept.'}}))
    predictions = [
        {'stock_name': 'AAA', 'company_name': 'A AG', 'company_description': 'Makes things.'},
        {'stock_name': 'BBB', 'company_name': 'B AG', 'company_description': 'Error fetching description.'},
    ]
    assert update_profile_cache(predictions, str(cache)) == 2
    assert set(json.loads(cache.read_text())) == {'AAA', 'OLD'}
