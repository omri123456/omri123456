"""Compare a model-driven strategy with honest benchmarks over a period."""

from __future__ import annotations

import pandas as pd

from ..features import rules
from ..features.panel import PanelBundle
from . import metrics
from .backtest import simulate
from .strategy import target_weights, trend_rule


def _summary(name: str, equity: pd.DataFrame, trades: pd.DataFrame, bundle: PanelBundle) -> dict:
    stats = {"strategy": name}
    stats.update(metrics.performance(equity["equity"], bundle.cash_rate))
    stats.update(metrics.exposure_stats(equity["weight"], trades, equity["equity"]))
    next_day = bundle.price.pct_change().shift(-1)
    stats.update(metrics.timing_pvalue(equity["weight"], next_day))
    stats["costs_paid"] = equity.attrs.get("costs_paid", 0.0)
    return stats


def compare(pred: pd.DataFrame, bundle: PanelBundle, cfg: dict, *, label: str = "cnn",
            mode: str | None = None, start=None, end=None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    """Simulate the model strategy and the benchmarks on the prediction dates in [start, end].

    Returns (summary table, {name: equity table}, {name: trades}).
    """
    tcfg, lag = cfg["trading"], int(cfg["model"]["execution_lag"])
    horizon = int(cfg["model"]["trade_horizon"])
    # sizing uses the full prediction history (causal), then we cut the period
    weights_all = target_weights(pred, tcfg, horizon, mode)
    dates = pred.index
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    targets = {label: weights_all.reindex(dates)}
    targets["buy_and_hold"] = pd.Series(1.0, index=dates)
    trend = trend_rule(bundle.price).reindex(dates)
    targets["trend_200d"] = trend

    curves, trade_logs, rows = {}, {}, []
    for name, target in targets.items():
        equity, trades = simulate(bundle.price, target, bundle.cash_rate, tcfg, lag)
        curves[name], trade_logs[name] = equity, trades
        rows.append(_summary(name, equity, trades, bundle))
    # same average exposure as the model, held constantly: isolates the value of timing
    avg = float(curves[label]["weight"].mean())
    equity, trades = simulate(bundle.price, pd.Series(avg, index=dates), bundle.cash_rate,
                              {**tcfg, "rebalance_band": 0.05}, lag)
    curves["constant_mix"], trade_logs["constant_mix"] = equity, trades
    rows.append(_summary("constant_mix", equity, trades, bundle))
    return pd.DataFrame(rows).set_index("strategy"), curves, trade_logs


def yearly_returns(curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calendar-year returns of every curve, side by side."""
    out = {}
    for name, equity in curves.items():
        series = equity["equity"]
        year_end = series.groupby(series.index.year).last()
        year_start = series.groupby(series.index.year).first()
        previous = year_end.shift(1).fillna(year_start)
        out[name] = year_end / previous - 1.0
    return pd.DataFrame(out)


def classic_rules(bundle: PanelBundle, cfg: dict, *, start=None, end=None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run every classic rule through the same broker, costs and execution delay as the model."""
    tcfg, lag = cfg["trading"], int(cfg["model"]["execution_lag"])
    signals = rules.rule_signals(bundle.price)  # computed on the full history, then cut
    dates = bundle.price.index
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    rows, curves = [], {}
    for name in signals.columns:
        target = signals[name].reindex(dates).fillna(1.0).clip(0.0, float(tcfg["max_weight"]))
        equity, trades = simulate(bundle.price, target, bundle.cash_rate, tcfg, lag)
        curves[name] = equity
        rows.append(_summary(name, equity, trades, bundle))
    return pd.DataFrame(rows).set_index("strategy"), curves
