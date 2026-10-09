import numpy as np
import pandas as pd

from config.settings import load_settings
from src.processing.features import engineer_features


def test_load_settings_returns_independent_copies():
    first = load_settings()
    first['horizon_days'] = -1
    assert load_settings()['horizon_days'] != -1


def test_settings_contain_parameters_used_by_the_pipeline():
    settings = load_settings()
    for key in ('horizon_days', 'threshold', 'features_to_select', 'anova_k',
                'step1_history_years', 'feature_warmup_years', 'quantiles',
                'min_cv_accuracy', 'min_step2_score'):
        assert key in settings
    assert 'step2_history_years' not in settings


def test_engineer_features_trims_warmup_history_and_labels_target():
    index = pd.bdate_range('2014-01-01', '2026-01-01')
    close = pd.Series(np.linspace(100, 300, len(index)), index=index)
    df = pd.DataFrame({'Close': close, 'Volume': 1.0})

    out = engineer_features(df, horizon_days=126, threshold=0.10, history_years=10)

    assert out.index.min() >= index.max() - pd.DateOffset(years=10)
    # Warm-up history keeps the 252-day momentum populated from the first kept row
    assert out['Ret_252D'].notna().all()
    # The last `horizon_days` rows have no observable target
    assert out['Target'].iloc[-126:].isna().all()
    assert out['Target'].iloc[:-126].notna().all()
    assert 'Close' not in out.columns
