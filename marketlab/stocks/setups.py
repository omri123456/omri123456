"""The day-trader scanner's daily rules, computed for every stock on every past day, and the
market-breadth series that come out of them.

The rules mirror trader-scanner/scan.py `find_setups` exactly, on closes:
  breakout_20d     the close is above every close of the 20 sessions before it
  bull_flag        up more than 10% over the 10 sessions before the last 5, then the last 5
                   closes inside a 5% range
  reclaim_ma20     the close before was under its 20-day average, this close is above it
  oversold         RSI 14 (Wilder) under 30
  oversold_bounce  RSI 14 was under 30 the day before and today closed up
The scanner seeds RSI from a 3-month window; here it runs over the whole history, so RSI can
differ by a fraction of a point in the first weeks of a stock's life and not after.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SETUPS = ["oversold", "oversold_bounce", "breakout_20d", "bull_flag", "reclaim_ma20"]


def wilder_rsi(closes: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    change = closes.diff()
    gain = change.clip(lower=0.0)
    loss = (-change).clip(lower=0.0)
    out = {}
    for col in closes.columns:
        g, l = gain[col].to_numpy(), loss[col].to_numpy()
        valid = np.flatnonzero(~np.isnan(g))
        series = np.full(len(g), np.nan)
        if len(valid) > n:
            first = valid[0]
            ag, al = np.mean(g[first:first + n]), np.mean(l[first:first + n])
            for i in range(first + n, len(g)):
                if np.isnan(g[i]):  # a gap (holiday for this stock): carry the averages
                    series[i] = series[i - 1]
                    continue
                if i > first + n - 1:
                    ag = (ag * (n - 1) + g[i]) / n
                    al = (al * (n - 1) + l[i]) / n
                series[i] = 100.0 - 100.0 / (1.0 + ag / al) if al > 0 else 100.0
        out[col] = series
    return pd.DataFrame(out, index=closes.index)


def _flags_one(c: pd.Series) -> dict[str, pd.Series]:
    """Rules for one stock on its own trading days (missing days dropped, as the scanner does)."""
    c = c.dropna().astype(float)
    frame = c.to_frame("c")
    enough = c.rolling(30, min_periods=30).count() >= 30
    hi20 = c.shift(1).rolling(20, min_periods=20).max()
    ma20 = c.rolling(20, min_periods=20).mean()
    ma20_prev = c.shift(1).rolling(20, min_periods=20).mean()
    run = c.shift(5) / c.shift(15) - 1.0
    rng5 = (c.rolling(5, min_periods=5).max() - c.rolling(5, min_periods=5).min()) / c
    rsi = wilder_rsi(frame)["c"]
    out = {
        "breakout_20d": c > hi20,
        "bull_flag": (run > 0.10) & (rng5 < 0.05),
        "reclaim_ma20": (c.shift(1) < ma20_prev) & (c > ma20),
        "oversold": rsi < 30,
        "oversold_bounce": (rsi.shift(1) < 30) & (c > c.shift(1)),
    }
    out = {k: (v & enough) for k, v in out.items()}
    out["_rsi"] = rsi
    return out


def setup_flags(closes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One boolean frame (dates x symbols) per rule; False where a stock has < 30 closes."""
    per = {sym: _flags_one(closes[sym]) for sym in closes.columns}
    result = {}
    for key in SETUPS + ["_rsi"]:
        frame = pd.DataFrame({sym: per[sym][key] for sym in closes.columns}).reindex(closes.index)
        result[key] = frame if key == "_rsi" else frame.fillna(False).astype(bool)
    return result


def breadth(closes: pd.DataFrame, flags: dict[str, pd.DataFrame] | None = None, min_members: int = 40) -> pd.DataFrame:
    """Daily market breadth among the members that traded that day (NaN with fewer than min_members)."""
    c = closes.astype(float)
    flags = flags or setup_flags(c)
    live = c.notna() & (c.rolling(30, min_periods=30).count() >= 30)
    n = live.sum(axis=1).astype(float)
    out = pd.DataFrame(index=c.index)
    for name in SETUPS:
        out[f"br_{name}"] = (flags[name] & live).sum(axis=1) / n
    for window in (50, 200):
        ma = c.rolling(window, min_periods=window).mean()
        has = live & ma.notna()
        out[f"br_above_ma{window}"] = ((c > ma) & has).sum(axis=1) / has.sum(axis=1).replace(0, np.nan)
    ret = c.pct_change(fill_method=None)
    adv = ((ret > 0) & live).sum(axis=1) - ((ret < 0) & live).sum(axis=1)
    out["br_adv_dec_10d"] = (adv / n).rolling(10, min_periods=5).mean()
    hi = c.rolling(252, min_periods=200).max()
    lo = c.rolling(252, min_periods=200).min()
    has = live & hi.notna()
    out["br_new_high_low"] = (((c >= hi) & has).sum(axis=1) - ((c <= lo) & has).sum(axis=1)) / has.sum(axis=1).replace(0, np.nan)
    out["br_members"] = n
    out.loc[n < min_members, [col for col in out.columns if col != "br_members"]] = np.nan
    return out
