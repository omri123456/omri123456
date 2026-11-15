"""Classic rule-based trading methods on the index, as daily target weights in [0, 1].

Each rule answers "how much of the account should be in the index after today's close?"
using closing prices up to today only. They serve two purposes: honest competitors for the
CNN in the same virtual-money simulation, and optional extra inputs for the CNN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RULES = {
    "trend_200d": "מגמה: מעל ממוצע 200 יום",
    "ma_cross_50_200": "חציית ממוצעים 50/200",
    "momentum_12m": "מומנטום 12 חודשים",
    "breakout_55_20": "פריצה: שיא 55 יום, יציאה בשפל 20 יום",
    "macd": "MACD (12, 26, 9)",
    "rsi2_reversion": "היפוך RSI(2) בתוך מגמת עלייה",
    "bollinger_reversion": "היפוך מרצועת בולינגר התחתונה",
    "vol_target_15": "יעד תנודתיות 15%",
    "turn_of_month": "אפקט סוף-חודש",
}


def _rsi(price: pd.Series, n: int) -> pd.Series:
    change = price.diff()
    gain = change.clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    loss = (-change.clip(upper=0.0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return 100.0 - 100.0 / (1.0 + gain / loss.replace(0.0, np.nan))


def _stateful(enter: np.ndarray, leave: np.ndarray) -> np.ndarray:
    """Position that switches on at an entry signal and off at an exit signal."""
    out = np.zeros(len(enter))
    held = False
    for i in range(len(enter)):
        if held and leave[i]:
            held = False
        elif not held and enter[i]:
            held = True
        out[i] = 1.0 if held else 0.0
    return out


def macd_lines(price: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD as continuous numbers, scaled by price so they compare across decades.

    macd      = EMA(fast) - EMA(slow): the strength and direction of the trend
    histogram = macd - its own EMA(signal): whether that trend is gaining or losing speed
    """
    price = price.astype(float)
    quick = price.ewm(span=fast, adjust=False, min_periods=slow).mean()
    lazy = price.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = (quick - lazy) / price
    line = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": macd, "macd_hist": macd - line})


def rule_signals(price: pd.Series) -> pd.DataFrame:
    """Target weight of every rule on every trading day (NaN until a rule has enough history)."""
    price = price.astype(float)
    ret = np.log(price).diff()
    ma5, ma20, ma50, ma200 = (price.rolling(n, min_periods=n).mean() for n in (5, 20, 50, 200))
    out = pd.DataFrame(index=price.index)

    out["trend_200d"] = (price > ma200).astype(float).where(ma200.notna())
    out["ma_cross_50_200"] = (ma50 > ma200).astype(float).where(ma200.notna())
    past = price.shift(252)
    out["momentum_12m"] = (price > past).astype(float).where(past.notna())

    high55 = price.rolling(55, min_periods=55).max()
    low20 = price.rolling(20, min_periods=20).min()
    breakout = _stateful((price >= high55).to_numpy(), (price <= low20).to_numpy())
    out["breakout_55_20"] = pd.Series(breakout, index=price.index).where(high55.notna())

    fast = price.ewm(span=12, adjust=False, min_periods=26).mean()
    slow = price.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd"] = (macd > signal).astype(float).where(signal.notna())

    rsi2 = _rsi(price, 2)
    dip = ((rsi2 < 10.0) & (price > ma200)).to_numpy()
    recovered = ((rsi2 > 70.0) | (price > ma5)).to_numpy()
    out["rsi2_reversion"] = pd.Series(_stateful(dip, recovered), index=price.index).where(ma200.notna())

    band = price.rolling(20, min_periods=20).std()
    below = (price < ma20 - 2.0 * band).to_numpy()
    back = (price >= ma20).to_numpy()
    out["bollinger_reversion"] = pd.Series(_stateful(below, back), index=price.index).where(band.notna())

    realized = ret.rolling(20, min_periods=20).std() * np.sqrt(252.0)
    out["vol_target_15"] = (0.15 / realized).clip(upper=1.0)

    # Turn of the month: be invested for the last trading day and the first three of the next
    # month. An order decided after today's close is filled at tomorrow's close, so the signal
    # switches on two business days before month end and off after the first one.
    days = price.index.values.astype("datetime64[D]")
    month_end = (price.index + pd.offsets.BMonthEnd(0)).values.astype("datetime64[D]")
    month_start = price.index.to_period("M").to_timestamp().values.astype("datetime64[D]")
    to_end = np.busday_count(days, month_end)
    from_start = np.busday_count(month_start, days)
    out["turn_of_month"] = ((to_end <= 2) | (from_start <= 0)).astype(float)
    return out[list(RULES)]
