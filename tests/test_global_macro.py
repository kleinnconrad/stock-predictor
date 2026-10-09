import numpy as np
import pandas as pd

from src.ingestion import global_macro
from src.ingestion.global_macro import (
    align_to_business_days,
    apply_publication_lag,
    infer_observation_frequency,
)


def _calendar_frame():
    # Calendar index including weekends, as produced by joining BTC-USD with other series
    index = pd.date_range('2026-01-01', '2026-01-31', freq='D')
    return pd.DataFrame({'BTC-USD': np.arange(len(index), dtype=float)}, index=index)


def test_weekend_dated_series_is_carried_to_business_days():
    df = _calendar_frame()
    df['ICSA'] = np.nan
    df.loc['2026-01-03', 'ICSA'] = 200.0  # Saturday
    df.loc['2026-01-10', 'ICSA'] = 210.0  # Saturday

    out = align_to_business_days(df, end_date='2026-01-30')

    assert out.index.dayofweek.max() <= 4
    assert out.loc['2026-01-05', 'ICSA'] == 200.0
    assert out.loc['2026-01-09', 'ICSA'] == 200.0
    assert out.loc['2026-01-12', 'ICSA'] == 210.0
    assert out.loc['2026-01-30', 'ICSA'] == 210.0


def test_holiday_gaps_in_a_column_are_forward_filled():
    df = _calendar_frame()
    df['SPY'] = 100.0
    df.loc['2026-01-19', 'SPY'] = np.nan  # US holiday, other markets open

    out = align_to_business_days(df, end_date='2026-01-30')

    assert out['SPY'].notna().all()


def test_rows_after_end_date_are_dropped():
    out = align_to_business_days(_calendar_frame(), end_date='2026-01-15')
    assert out.index.max() == pd.Timestamp('2026-01-15')


LAGS = {'daily': 1, 'weekly': 7, 'monthly': 60, 'quarterly': 150}


def test_monthly_observations_are_moved_to_their_publication_date():
    monthly = pd.Series([300.0, 301.0, 302.0],
                        index=pd.to_datetime(['2026-01-01', '2026-02-01', '2026-03-01']))
    out = apply_publication_lag(pd.DataFrame({'CPIAUCSL': monthly}), LAGS)

    assert list(out['CPIAUCSL'].dropna().index) == list(monthly.index + pd.Timedelta(days=60))


def test_each_series_gets_the_lag_of_its_own_frequency():
    daily_index = pd.bdate_range('2026-01-01', '2026-03-31')
    weekly_index = pd.date_range('2026-01-03', '2026-03-28', freq='W-SAT')
    fred_df = pd.concat([
        pd.Series(1.0, index=daily_index, name='T10Y2Y'),
        pd.Series(2.0, index=weekly_index, name='ICSA'),
    ], axis=1, sort=True)

    out = apply_publication_lag(fred_df, LAGS)

    assert out['T10Y2Y'].dropna().index.min() == daily_index.min() + pd.Timedelta(days=1)
    assert out['ICSA'].dropna().index.min() == weekly_index.min() + pd.Timedelta(days=7)


def test_infer_observation_frequency():
    assert infer_observation_frequency(pd.bdate_range('2026-01-01', periods=30)) == 'daily'
    assert infer_observation_frequency(pd.date_range('2026-01-01', periods=10, freq='W')) == 'weekly'
    assert infer_observation_frequency(pd.date_range('2026-01-01', periods=10, freq='MS')) == 'monthly'
    assert infer_observation_frequency(pd.date_range('2026-01-01', periods=10, freq='QS')) == 'quarterly'


def test_fred_fetch_skips_failing_and_discontinued_series(monkeypatch):
    def fake_reader(series_id, source, start, end, api_key=None):
        if series_id == 'BROKEN':
            raise IOError('not a valid series')
        last = '2026-09-01' if series_id == 'FRESH' else '2021-06-01'
        index = pd.date_range(end=last, periods=24, freq='MS')
        return pd.DataFrame({series_id: np.arange(24, dtype=float)}, index=index)

    monkeypatch.setattr(global_macro.web, 'DataReader', fake_reader)

    out = global_macro.fetch_fred_indicators(
        ['FRESH', 'BROKEN', 'DISCONTINUED'], '2020-01-01', '2026-10-01', max_staleness_days=400)

    assert list(out.columns) == ['FRESH']


def test_universe_contains_documented_commodities_and_no_discontinued_series():
    from config.universe import ALL_FRED_INDICATORS, ALL_YF_TICKERS

    for ticker in ('CL=F', 'GC=F', 'HG=F', 'ZC=F', 'ZW=F', 'LE=F', 'LBR=F'):
        assert ticker in ALL_YF_TICKERS
    assert 'OJ=F' not in ALL_YF_TICKERS
    for series_id in ('ECBASSETS', 'LRHUTTTTEZM156S', 'PRINTO01EZQ661S', 'JPNCPIALLMINMEI',
                      'JPNPROINDMISMEI', 'GBRCPIALLMINMEI', 'GBRPROINDMISMEI'):
        assert series_id not in ALL_FRED_INDICATORS
    assert len(ALL_FRED_INDICATORS) == len(set(ALL_FRED_INDICATORS))
