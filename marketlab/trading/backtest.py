"""Event loop shared by the backtest and the forward paper account.

On every trading day, in this order:
  1. cash earns interest for the days since the previous mark;
  2. the order decided `execution_lag` rows earlier is filled at today's close (if the target
     moved by more than the rebalance band);
  3. the account is marked to today's close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .broker import PaperBroker


def step(broker: PaperBroker, date: pd.Timestamp, price: float, cash_rate: float,
         target: float | None, band: float, max_weight: float) -> dict:
    """Process one close. `target` is the weight decided earlier for execution today (or None)."""
    stamp = str(pd.Timestamp(date).date())
    if broker.last_date is not None:
        days = (pd.Timestamp(date) - pd.Timestamp(broker.last_date)).days
        broker.accrue_interest(cash_rate, days)
    traded = None
    if target is not None and np.isfinite(target):
        current = broker.weight(price)
        at_edge = target in (0.0, max_weight) and abs(target - current) > 1e-3
        if abs(target - current) > band or at_edge:
            traded = broker.rebalance_to(float(target), price, stamp)
    equity = broker.mark(stamp, price)
    return {"date": pd.Timestamp(date), "equity": equity, "weight": broker.weight(), "price": float(price),
            "cash": broker.cash, "traded": traded is not None}


def simulate(price: pd.Series, targets: pd.Series, cash_rate: pd.Series, tcfg: dict,
             execution_lag: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a strategy over `targets.index`. Returns (daily equity table, trade list).

    targets[t] is the weight decided with information of row t; it is filled at the close of
    row t + execution_lag. The account starts in cash on the first day.
    """
    dates = targets.index
    price = price.reindex(dates).astype(float)
    cash_rate = cash_rate.reindex(dates).ffill().fillna(0.0)
    pending = targets.shift(execution_lag)
    broker = PaperBroker(cash=float(tcfg["initial_cash"]), cost_bps=float(tcfg["cost_bps"]))
    band, max_weight = float(tcfg["rebalance_band"]), float(tcfg["max_weight"])
    rows = []
    for date, px, rate, target in zip(dates, price.to_numpy(), cash_rate.to_numpy(), pending.to_numpy()):
        rows.append(step(broker, date, px, rate, None if np.isnan(target) else target, band, max_weight))
    equity = pd.DataFrame(rows).set_index("date")
    trades = pd.DataFrame(broker.trades)
    equity.attrs["costs_paid"] = broker.costs_paid
    equity.attrs["interest_earned"] = broker.interest_earned
    return equity, trades
