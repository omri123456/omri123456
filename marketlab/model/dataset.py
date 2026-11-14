"""Labels, scaling and windowing.

Timing: the sample at row t holds features known for an order filled at the close of row
t + execution_lag. Its label for horizon h is the log return from that fill to h days later:
    y_t(h) = log P[t + lag + h] - log P[t + lag]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class Labels:
    ret: np.ndarray   # [T, H] forward log return, NaN where the future is unknown
    z: np.ndarray     # [T, H] forward return divided by the volatility known at t
    up: np.ndarray    # [T, H] 1.0 if ret > 0, 0.0 otherwise, NaN where unknown
    horizons: list[int]
    span: int         # rows of future prices the longest label needs (lag + max horizon)


def make_labels(price: pd.Series, horizons: list[int], execution_lag: int) -> Labels:
    logp = np.log(price.astype(float))
    daily = logp.diff()
    vol = daily.rolling(20, min_periods=10).std().clip(lower=0.004)  # known at t
    ret = np.full((len(price), len(horizons)), np.nan)
    z = np.full_like(ret, np.nan)
    for j, h in enumerate(horizons):
        fwd = logp.shift(-(execution_lag + h)) - logp.shift(-execution_lag)
        ret[:, j] = fwd.to_numpy()
        z[:, j] = (fwd / (vol * np.sqrt(h))).to_numpy()
    up = np.where(np.isnan(ret), np.nan, (ret > 0).astype(float))
    return Labels(ret=ret, z=z, up=up, horizons=list(horizons), span=execution_lag + max(horizons))


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray
    clip: float

    @classmethod
    def fit(cls, values: np.ndarray, clip: float) -> "Scaler":
        """Column statistics from the training rows only (NaN-aware)."""
        with np.errstate(all="ignore"):
            mean = np.nanmean(values, axis=0)
            std = np.nanstd(values, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
        return cls(mean=mean, std=std, clip=clip)

    def transform(self, values: np.ndarray) -> np.ndarray:
        scaled = (values - self.mean) / self.std
        scaled = np.clip(scaled, -self.clip, self.clip)
        return np.nan_to_num(scaled, nan=0.0).astype(np.float32)  # missing -> the training mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "clip": self.clip}

    @classmethod
    def from_dict(cls, d: dict) -> "Scaler":
        return cls(mean=np.asarray(d["mean"]), std=np.asarray(d["std"]), clip=float(d["clip"]))


def make_windows(matrix: np.ndarray, rows: np.ndarray, window: int) -> torch.Tensor:
    """Stack windows ending at each row: returns [N, C, window] (channels first)."""
    if len(rows) == 0:
        return torch.zeros((0, matrix.shape[1], window), dtype=torch.float32)
    if rows.min() < window - 1:
        raise ValueError("a window would start before the first panel row")
    idx = rows[:, None] + np.arange(-window + 1, 1)[None, :]
    return torch.from_numpy(np.ascontiguousarray(matrix[idx].transpose(0, 2, 1)))


def train_rows_before(first_test_row: int, window: int, label_span: int) -> np.ndarray:
    """Rows usable for training when testing starts at first_test_row.

    The last training label must be fully realised before the first test decision, so we stop
    label_span rows early (purging): row i needs prices up to i + label_span.
    """
    return np.arange(window - 1, max(window - 1, first_test_row - label_span))
