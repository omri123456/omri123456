"""Point-in-time alignment helpers.

Convention: row t of the panel holds what can be known in time to place an order that is
filled at the close of trading day t+1. Every helper here is causal: the value written in row t
only depends on observations that were already published by then.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_daily(series: pd.Series, calendar: pd.DatetimeIndex, lag: int, *, limit: int = 5) -> pd.Series:
    """Daily market series: take the latest observation on or before each trading day (at most
    `limit` rows stale), then delay by `lag` trading days to honour the publication delay."""
    aligned = series.reindex(calendar, method="ffill", limit=limit)
    return aligned.shift(lag) if lag else aligned


def align_available(series: pd.Series | pd.DataFrame, available: pd.DatetimeIndex,
                    calendar: pd.DatetimeIndex, *, limit: int | None = None):
    """As-of join on availability dates: row t gets the newest observation published by t."""
    obj = series.copy()
    obj.index = pd.DatetimeIndex(available)
    obj = obj[~obj.index.duplicated(keep="last")].sort_index()
    return obj.reindex(calendar, method="ffill", limit=limit)


def available_dates(index: pd.DatetimeIndex, freq: str, lag: int) -> pd.DatetimeIndex:
    """Date by which an observation is published.

    freq 'w' / 'c': lag in calendar days after the observation date.
    freq 'm': lag in calendar days after the end of the reference month.
    """
    if freq == "m":
        return index + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag)
    return index + pd.Timedelta(days=lag)


def bucket_to_calendar(daily: pd.DataFrame, calendar: pd.DatetimeIndex, how: str = "sum") -> pd.DataFrame:
    """Aggregate calendar-day data (weekends included) onto trading days.

    Each calendar day is assigned to the first trading day on or after it, so Saturday and
    Sunday news lands on Monday's row. Days after the last trading day are dropped.
    """
    if daily.empty:
        return pd.DataFrame(index=calendar, columns=daily.columns, dtype=float)
    positions = calendar.searchsorted(daily.index, side="left")
    keep = positions < len(calendar)
    grouped = daily[keep].groupby(calendar[positions[keep]])
    out = grouped.sum() if how == "sum" else grouped.mean()
    return out.reindex(calendar)


def zscore(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    """Trailing z-score: today's value against its own recent history (includes today)."""
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0.0, np.nan)


def surprise(series: pd.Series, window: int = 250, min_periods: int = 60, floor: float = 0.25) -> pd.Series:
    """How unusual today's value is against the window that ended yesterday.

    Used for sparse count-like series; `floor` keeps quiet histories from producing huge values.
    """
    history = series.shift(1)
    mean = history.rolling(window, min_periods=min_periods).mean()
    std = history.rolling(window, min_periods=min_periods).std()
    return (series - mean) / (std + floor)


def log_safe(series: pd.Series, floor: float = 1e-6) -> pd.Series:
    return np.log(series.clip(lower=floor))
