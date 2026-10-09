import numpy as np
import pandas as pd
import pytest

from config.settings import load_settings
from src.modeling import step1_macro
from src.modeling.base_pipeline import InsufficientHistoryError, purged_time_series_cv


@pytest.fixture
def fast_settings(monkeypatch, tmp_path):
    settings = load_settings()
    settings.update({'features_to_select': 2, 'anova_k': 4, 'sfs_cv_splits': 3})
    monkeypatch.setattr(step1_macro, 'load_settings', lambda: dict(settings))
    monkeypatch.chdir(tmp_path)  # diagnostic artifacts are written relative to the cwd
    return settings


def _frame(n_rows, target, horizon=126, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range('2014-01-01', periods=n_rows)
    df = pd.DataFrame(rng.normal(size=(n_rows, 6)), index=index, columns=[f'f{i}' for i in range(6)])
    df['Target'] = np.asarray(target, dtype=float)
    df['Future_Return'] = 0.0
    df.iloc[-horizon:, df.columns.get_loc('Target')] = np.nan
    df.iloc[-horizon:, df.columns.get_loc('Future_Return')] = np.nan
    return df


@pytest.mark.parametrize('n_samples,n_splits', [(2400, 2), (1500, 5), (900, 3)])
def test_purged_splits_never_overlap_the_label_horizon(n_samples, n_splits):
    cv, purged = purged_time_series_cv(n_samples, n_splits, gap=126, min_train_rows=126)
    assert purged
    for train_index, test_index in cv.split(np.zeros(n_samples)):
        assert train_index.max() + 126 < test_index.min()
        assert len(train_index) >= 126


def test_purged_split_raises_or_falls_back_when_history_is_too_short():
    with pytest.raises(InsufficientHistoryError):
        purged_time_series_cv(500, 2, gap=126, min_train_rows=126)
    cv, purged = purged_time_series_cv(500, 5, gap=126, min_train_rows=126, allow_unpurged=True)
    assert not purged and cv.n_splits == 5


def test_short_history_is_rejected_without_invented_scores(fast_settings):
    metrics = step1_macro.execute_step1(_frame(600, np.tile([0, 1], 300)), ticker='SHORT')
    assert metrics['cv_status'] == 'insufficient_history'
    assert metrics['predicted_class'] == 'NOT_UP'
    assert metrics['cv_accuracy'] is None


def test_single_class_training_fold_is_rejected_instead_of_scored_in_sample(fast_settings):
    # Positives only appear late in the history, so the first training fold has one class
    target = np.zeros(2000)
    target[1700:] = 1
    metrics = step1_macro.execute_step1(_frame(2000, target), ticker='RARE')
    assert metrics['cv_status'].startswith('cv_failed')
    assert metrics['predicted_class'] == 'NOT_UP'
    assert metrics['cv_accuracy'] is None


def test_single_class_target_is_rejected(fast_settings):
    metrics = step1_macro.execute_step1(_frame(2000, np.zeros(2000)), ticker='FLAT')
    assert metrics['cv_status'] == 'single_class_target'


def test_learnable_signal_is_validated(fast_settings):
    rng = np.random.default_rng(1)
    df = _frame(2000, np.zeros(2000))
    df['Target'] = np.where(df['f0'] + 0.3 * rng.normal(size=2000) > 0, 1.0, 0.0)
    df.iloc[-126:, df.columns.get_loc('Target')] = np.nan

    metrics = step1_macro.execute_step1(df, ticker='SIGNAL')

    assert metrics['cv_status'] == 'ok'
    assert metrics['cv_accuracy'] > 0.7
    assert 'f0' in metrics['selected_predictors_and_weights']
    assert metrics['predicted_class'] in ('UP', 'NOT_UP')
