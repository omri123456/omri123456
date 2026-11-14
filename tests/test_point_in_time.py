"""The properties that make a backtest trustworthy: no value may be used before it is known."""

import numpy as np
import pandas as pd

from marketlab.features.align import align_daily, available_dates, bucket_to_calendar, surprise
from marketlab.features.panel import build_panel
from marketlab.model.dataset import make_labels, train_rows_before


def test_panel_never_uses_observations_from_the_future(raw, cfg, bundle):
    """Rebuilding the panel from data cut at day T must reproduce every row up to T exactly."""
    for cut in ("2007-06-29", "2009-03-10", "2010-11-01"):
        truncated = build_panel(raw.truncate(cut), cfg).panel
        full = bundle.panel.loc[:cut]
        assert list(truncated.columns) == list(full.columns)
        assert truncated.index.equals(full.index)
        np.testing.assert_allclose(truncated.to_numpy(), full.to_numpy(), rtol=1e-9, atol=1e-9, equal_nan=True)


def test_monthly_series_respect_their_publication_delay(raw, cfg, bundle):
    """Changing a monthly number must not move any row dated before its release."""
    changed = raw.truncate("2011-12-30")
    series = changed.fred["UNRATE"].copy()
    obs_date = series.index[-3]
    series.iloc[-3] *= 1.5
    changed.fred["UNRATE"] = series
    release = available_dates(pd.DatetimeIndex([obs_date]), "m", cfg["fred"]["UNRATE"]["lag"])[0]
    after = build_panel(changed, cfg).panel
    before_release = bundle.panel.index < release
    np.testing.assert_allclose(after.loc[before_release, "unrate_chg_3m"].to_numpy(),
                               bundle.panel.loc[before_release, "unrate_chg_3m"].to_numpy(), equal_nan=True)
    assert not np.allclose(after.loc[~before_release, "unrate_chg_3m"].to_numpy(),
                           bundle.panel.loc[~before_release, "unrate_chg_3m"].to_numpy(), equal_nan=True)
    assert release > obs_date + pd.offsets.MonthEnd(0)


def test_daily_series_are_delayed_by_their_lag(raw, cfg, bundle):
    """VIX is published with a one-day delay: day d's value first appears in row d+1."""
    changed = raw.truncate("2011-12-30")
    vix = changed.fred["VIXCLS"].copy()
    day = vix.index[-10]
    vix.loc[day] *= 3.0
    changed.fred["VIXCLS"] = vix
    after = build_panel(changed, cfg).panel
    same = after["vix_log"].to_numpy() == bundle.panel["vix_log"].to_numpy()
    first_changed = after.index[np.flatnonzero(~same)[0]]
    assert first_changed > day
    assert first_changed == after.index[after.index.get_loc(day) + cfg["fred"]["VIXCLS"]["lag"]]


def test_align_daily_shifts_by_trading_rows():
    calendar = pd.bdate_range("2020-01-01", periods=6)
    series = pd.Series(np.arange(6.0), index=calendar)
    assert align_daily(series, calendar, 0).tolist() == [0, 1, 2, 3, 4, 5]
    lagged = align_daily(series, calendar, 2)
    assert np.isnan(lagged.iloc[:2]).all() and lagged.iloc[2:].tolist() == [0, 1, 2, 3]


def test_weekend_news_lands_on_the_next_trading_day():
    calendar = pd.DatetimeIndex(["2020-01-03", "2020-01-06", "2020-01-07"])  # Fri, Mon, Tue
    daily = pd.DataFrame({"n": [1.0, 2.0, 4.0, 8.0, 16.0]},
                         index=pd.date_range("2020-01-03", "2020-01-07", freq="D"))
    bucket = bucket_to_calendar(daily, calendar, "sum")
    assert bucket["n"].tolist() == [1.0, 2.0 + 4.0 + 8.0, 16.0]


def test_surprise_only_looks_backwards():
    values = pd.Series(np.r_[np.zeros(100), 10.0, np.zeros(5)])
    score = surprise(values, window=50, min_periods=20, floor=0.25)
    assert score.iloc[100] > 5          # the spike is a surprise on its own day
    assert abs(score.iloc[99]) < 1e-9   # and invisible the day before


def test_labels_are_the_return_after_the_fill(cfg):
    price = pd.Series(np.exp(np.linspace(0, 1, 50)), index=pd.bdate_range("2020-01-01", periods=50))
    labels = make_labels(price, [1, 5], execution_lag=1)
    logp = np.log(price.to_numpy())
    t = 10
    assert np.isclose(labels.ret[t, 0], logp[t + 2] - logp[t + 1])
    assert np.isclose(labels.ret[t, 1], logp[t + 6] - logp[t + 1])
    assert np.isnan(labels.ret[-1]).all() and np.isnan(labels.ret[-6, 1])
    assert labels.span == 6


def test_training_rows_are_purged_before_the_test_period():
    rows = train_rows_before(first_test_row=1000, window=60, label_span=21)
    # the last training label ends at row max + span, which must be before the first test row
    assert rows.max() + 21 < 1000
    assert rows.min() == 59
