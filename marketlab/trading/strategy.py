"""Turn model output into a target portfolio weight.

Two sizing rules, both anchored on "no information => stay invested like buy and hold":

absolute  the model's probability against the long-run base rate. p = base rate -> fully
          invested, p = 50% -> out. Uses the probability level, so a model that is biased in a
          new regime shifts the whole exposure.
relative  the model's probability against the same model's predictions over the previous year
          (stored with every prediction as ref_mean / ref_std). Only asks "is the model more or
          less optimistic than usual?", which is robust to a biased level.

Both are causal: they only use predictions made up to that day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def relative_score(pred: pd.DataFrame, horizon: int) -> pd.Series:
    """How optimistic the model is today against its own recent predictions, in standard deviations."""
    spread = pred[f"ref_std_{horizon}"].astype(float).replace(0.0, np.nan)
    return ((pred[f"p_{horizon}"].astype(float) - pred[f"ref_mean_{horizon}"].astype(float)) / spread
            ).rename("score")


def target_weights(pred: pd.DataFrame, tcfg: dict, horizon: int, mode: str | None = None) -> pd.Series:
    """Target weight decided on each row (to be executed execution_lag rows later)."""
    mode = mode or tcfg.get("sizing", "relative")
    gain = float(tcfg.get("gain", 1.0))
    max_weight = float(tcfg["max_weight"])
    floor = -max_weight if tcfg.get("allow_short") else 0.0
    smooth = int(tcfg.get("signal_smooth", 1))

    if mode == "absolute":
        base = pred[f"base_p_{horizon}"].astype(float)
        signal = (pred[f"p_{horizon}"].astype(float) - base) / (base - 0.5).clip(lower=0.02)
    elif mode == "relative":
        signal = 0.5 * relative_score(pred, horizon)  # one standard deviation below usual -> half out
    else:
        raise ValueError(f"unknown sizing mode {mode!r}")
    if smooth > 1:
        signal = signal.rolling(smooth, min_periods=1).mean()
    weight = (1.0 + gain * signal).fillna(1.0)  # no reference yet: stay invested
    return weight.clip(lower=floor, upper=max_weight).rename("target_weight")


def trend_rule(price: pd.Series, window: int = 200) -> pd.Series:
    """Classic timing benchmark: invested while the index is above its moving average."""
    average = price.rolling(window, min_periods=window).mean()
    return (price > average).astype(float).where(average.notna(), 1.0).rename("target_weight")
