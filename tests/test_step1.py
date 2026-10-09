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


def test_cutoff_is_learned_on_earlier_folds_only():
    from src.modeling.diagnostics import walk_forward_predictions

    # Fold 1 separates at ~0.3; fold 2 would be perfectly separated at 0.75 if tuned on itself
    fold1 = (np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.4, 0.5]))
    fold2 = (np.array([0, 0, 1, 1]), np.array([0.6, 0.7, 0.8, 0.9]))

    y_true, y_pred = walk_forward_predictions([fold1, fold2])

    assert list(y_true) == [0, 0, 1, 1]
    assert list(y_pred) == [1, 1, 1, 1]  # cutoff from fold 1 (0.4) is applied, not re-tuned


def test_degenerate_cutoffs_are_detected():
    from src.modeling.diagnostics import ks_cutoff_or_none, walk_forward_predictions

    assert ks_cutoff_or_none(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])) is None
    # No discrimination: positives score lowest, so max(TPR - FPR) is reached only at +inf
    assert ks_cutoff_or_none(np.array([1, 1, 0, 0]), np.array([0.1, 0.2, 0.3, 0.4])) is None
    single_class_first_fold = (np.array([0, 0]), np.array([0.1, 0.2]))
    assert walk_forward_predictions([single_class_first_fold, single_class_first_fold]) is None
    cutoff = ks_cutoff_or_none(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert cutoff is not None and np.isfinite(cutoff[1])


def test_feature_names_stay_aligned_when_a_feature_has_no_observations(fast_settings):
    rng = np.random.default_rng(2)
    df = _frame(2000, np.zeros(2000))
    df['Target'] = np.where(df['f3'] + 0.3 * rng.normal(size=2000) > 0, 1.0, 0.0)
    df.iloc[-126:, df.columns.get_loc('Target')] = np.nan
    df.insert(0, 'empty_feature', np.nan)  # e.g. an instrument without history in the window

    metrics = step1_macro.execute_step1(df, ticker='EMPTY')

    assert metrics['cv_status'] == 'ok'
    assert 'f3' in metrics['selected_predictors_and_weights']
    assert 'empty_feature' not in metrics['selected_predictors_and_weights']


def test_failing_final_fit_is_reported_instead_of_raised(fast_settings, monkeypatch):
    from src.modeling import base_pipeline

    calls = {'n': 0}

    def build_failing_on_final_fit(*args, **kwargs):
        pipeline = base_pipeline.build_pipeline(*args, **kwargs)
        calls['n'] += 1
        if calls['n'] == 3:  # two CV folds succeed, the final fit on all rows fails
            def fail(*a, **k):
                raise ValueError("All the fits failed")
            pipeline.fit = fail
        return pipeline

    monkeypatch.setattr(step1_macro, 'build_pipeline', build_failing_on_final_fit)
    rng = np.random.default_rng(3)
    df = _frame(2000, np.zeros(2000))
    df['Target'] = np.where(df['f0'] + 0.3 * rng.normal(size=2000) > 0, 1.0, 0.0)
    df.iloc[-126:, df.columns.get_loc('Target')] = np.nan

    metrics = step1_macro.execute_step1(df, ticker='FINAL')

    assert metrics['cv_status'].startswith('cv_failed: final fit failed')
    assert metrics['predicted_class'] == 'NOT_UP'
