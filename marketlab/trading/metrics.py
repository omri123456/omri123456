"""Performance and prediction-quality statistics (plain numpy/pandas, no hidden assumptions)."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


def performance(equity: pd.Series, cash_rate: pd.Series | None = None) -> dict:
    """Summary of a daily equity curve. Sharpe is measured against the cash yield."""
    equity = equity.dropna()
    if len(equity) < 3:
        return {}
    daily = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    if cash_rate is not None:
        rf_daily = (cash_rate.reindex(daily.index).ffill().fillna(0.0) / 100.0) / TRADING_DAYS
    else:
        rf_daily = 0.0
    excess = daily - rf_daily
    vol = daily.std() * np.sqrt(TRADING_DAYS)
    sharpe = excess.mean() / excess.std() * np.sqrt(TRADING_DAYS) if excess.std() > 0 else np.nan
    drawdown = equity / equity.cummax() - 1.0
    max_dd = drawdown.min()
    return {
        "total_return": total, "cagr": cagr, "vol": vol, "sharpe": sharpe, "max_drawdown": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "worst_day": daily.min(), "best_day": daily.max(), "years": years,
        "final_equity": float(equity.iloc[-1]),
    }


def exposure_stats(weights: pd.Series, trades: pd.DataFrame, equity: pd.Series) -> dict:
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    traded = float(trades["value"].abs().sum()) if len(trades) else 0.0
    return {
        "avg_exposure": float(weights.mean()),
        "time_in_market": float((weights > 0.05).mean()),
        "trades_per_year": len(trades) / years,
        "turnover_per_year": traded / float(equity.mean()) / years,
    }


def timing_pvalue(weights: pd.Series, asset_returns: pd.Series, n_shifts: int = 2000, seed: int = 7) -> dict:
    """Is the timing better than the same exposure pattern applied at random times?

    We slide the whole weight series against the returns by a random circular offset. That
    keeps the average exposure and the persistence of both series, and destroys only their
    alignment. The p-value is the share of offsets that did at least as well as the real one.
    """
    w = weights.to_numpy(dtype=float)
    r = asset_returns.reindex(weights.index).fillna(0.0).to_numpy(dtype=float)
    n = len(w)
    if n < 250:
        return {"timing_pvalue": np.nan, "timing_gain": np.nan}

    def score(shifted: np.ndarray) -> float:
        return float(np.mean(shifted * r))

    actual = score(w)
    rng = np.random.default_rng(seed)
    offsets = rng.integers(20, n - 20, size=n_shifts)
    shuffled = np.array([score(np.roll(w, int(k))) for k in offsets])
    return {
        "timing_pvalue": float((np.sum(shuffled >= actual) + 1) / (n_shifts + 1)),
        # annualised return added by when the exposure was held, against random timing
        "timing_gain": float((actual - shuffled.mean()) * TRADING_DAYS),
    }


def spearman(x, y) -> float:
    """Rank correlation (average ranks for ties); NaN pairs are dropped. No scipy needed."""
    frame = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)}).dropna()
    if len(frame) < 3:
        return np.nan
    rx, ry = frame["x"].rank().to_numpy(), frame["y"].rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def auc(score: np.ndarray, outcome: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum identity (ties get average ranks)."""
    score, outcome = np.asarray(score, float), np.asarray(outcome, float)
    positives = outcome > 0.5
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def log_loss(prob: np.ndarray, outcome: np.ndarray) -> float:
    prob = np.clip(np.asarray(prob, float), 1e-6, 1 - 1e-6)
    outcome = np.asarray(outcome, float)
    return float(-np.mean(outcome * np.log(prob) + (1 - outcome) * np.log(1 - prob)))


def block_bootstrap_ci(values_fn, n: int, block: int = 20, draws: int = 500, seed: int = 3,
                       level: float = 0.90) -> tuple[float, float]:
    """Confidence interval for a statistic of time-ordered data, resampling blocks of days so
    that overlapping multi-day labels do not make the sample look larger than it is."""
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    stats = []
    for _ in range(draws):
        starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        value = values_fn(idx)
        if np.isfinite(value):
            stats.append(value)
    if len(stats) < 10:
        return (np.nan, np.nan)
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def prediction_quality(pred: pd.DataFrame, horizon: int) -> dict:
    """Out-of-sample quality of the direction forecast for one horizon."""
    frame = pred[[f"p_{horizon}", f"base_p_{horizon}", f"up_{horizon}", f"ret_{horizon}"]].dropna()
    if len(frame) < 50:
        return {}
    p = frame[f"p_{horizon}"].to_numpy()
    base = frame[f"base_p_{horizon}"].to_numpy()
    up = frame[f"up_{horizon}"].to_numpy()
    ret = frame[f"ret_{horizon}"].to_numpy()
    ll_model, ll_base = log_loss(p, up), log_loss(base, up)

    def ic_of(idx: np.ndarray) -> float:
        return spearman(p[idx], ret[idx])

    def skill_of(idx: np.ndarray) -> float:
        return 1.0 - log_loss(p[idx], up[idx]) / log_loss(base[idx], up[idx])

    everything = np.arange(len(p))
    block = max(20, 2 * horizon)
    ic_lo, ic_hi = block_bootstrap_ci(ic_of, len(p), block=block)
    sk_lo, sk_hi = block_bootstrap_ci(skill_of, len(p), block=block)
    score_ic = np.nan
    if f"ref_std_{horizon}" in pred.columns:  # the yardstick-relative score the strategy trades on
        rel = pred.loc[frame.index]
        spread = rel[f"ref_std_{horizon}"].replace(0.0, np.nan)
        score_ic = spearman((rel[f"p_{horizon}"] - rel[f"ref_mean_{horizon}"]) / spread, ret)
    return {
        "score_ic": score_ic,
        "n": len(frame), "up_rate": float(up.mean()), "mean_p": float(p.mean()),
        "accuracy": float(((p > 0.5) == (up > 0.5)).mean()),
        "always_up_accuracy": float(max(up.mean(), 1 - up.mean())),
        "auc": auc(p, up),
        "ic": ic_of(everything), "ic_lo": ic_lo, "ic_hi": ic_hi,
        "logloss": ll_model, "logloss_base": ll_base,
        "skill": 1.0 - ll_model / ll_base, "skill_lo": sk_lo, "skill_hi": sk_hi,
    }
