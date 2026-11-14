"""Forward paper account: virtual money that follows the model day by day.

The state lives in one JSON file so it survives between runs (and can be committed). Each
`advance` processes every trading day that appeared since the last run, using exactly the
event loop of the backtest: fill yesterday's decision at today's close, mark the account,
then record the new decision that will be filled at the next close.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.panel import PanelBundle
from .backtest import step
from .broker import PaperBroker
from .strategy import target_weights


def state_path(state_dir: Path) -> Path:
    return state_dir / "paper_account.json"


def load_state(state_dir: Path) -> dict | None:
    path = state_path(state_dir)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_state(state_dir: Path, state: dict) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    tmp = state_path(state_dir).with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=1)
    tmp.replace(state_path(state_dir))


def _decision(pred: pd.DataFrame, cfg: dict, date: pd.Timestamp) -> dict | None:
    """Target weight decided with the information of `date` (None if there is no prediction)."""
    if date not in pred.index:
        return None
    horizon = int(cfg["model"]["trade_horizon"])
    weights = target_weights(pred.loc[:date], cfg["trading"], horizon)
    value = float(weights.iloc[-1])
    if not np.isfinite(value):
        return None
    return {"decided_on": str(date.date()), "target_weight": round(value, 4),
            "p_up": round(float(pred.at[date, f"p_{horizon}"]), 4), "horizon": horizon}


def init_account(bundle: PanelBundle, pred: pd.DataFrame, cfg: dict, state_dir: Path,
                 start=None) -> dict:
    """Open the account in cash on `start` (default: the latest trading day in the data)."""
    dates = bundle.price.index
    first = dates[-1] if start is None else dates[dates.searchsorted(pd.Timestamp(start), side="left")]
    broker = PaperBroker(cash=float(cfg["trading"]["initial_cash"]), cost_bps=float(cfg["trading"]["cost_bps"]))
    broker.mark(str(first.date()), float(bundle.price.loc[first]))
    state = {
        "opened_on": str(first.date()),
        "instrument": cfg["project"]["target"],
        "initial_cash": float(cfg["trading"]["initial_cash"]),
        "broker": broker.to_dict(),
        "pending": _decision(pred, cfg, first),
        "history": [{"date": str(first.date()), "equity": broker.equity(), "weight": 0.0,
                     "price": float(bundle.price.loc[first]), "cash": broker.cash, "traded": False}],
    }
    state = advance(bundle, pred, cfg, state)  # catches up when start is in the past
    save_state(state_dir, state)
    return state


def advance(bundle: PanelBundle, pred: pd.DataFrame, cfg: dict, state: dict) -> dict:
    """Process all trading days after the account's last mark. Returns the updated state."""
    broker = PaperBroker.from_dict(state["broker"])
    tcfg = cfg["trading"]
    band, max_weight = float(tcfg["rebalance_band"]), float(tcfg["max_weight"])
    last = pd.Timestamp(broker.last_date)
    new_days = bundle.price.index[bundle.price.index > last]
    for date in new_days:
        pending = state.get("pending")
        target = pending["target_weight"] if pending and pd.Timestamp(pending["decided_on"]) < date else None
        rate = float(bundle.cash_rate.get(date, 0.0))
        row = step(broker, date, float(bundle.price.loc[date]), rate, target, band, max_weight)
        row["date"] = str(date.date())
        row["filled_decision"] = pending["decided_on"] if target is not None else None
        state["history"].append(row)
        state["pending"] = _decision(pred, cfg, date)
    state["broker"] = broker.to_dict()
    state["processed_days"] = len(new_days)
    return state


def summary(state: dict) -> dict:
    broker = PaperBroker.from_dict(state["broker"])
    equity = broker.equity()
    history = pd.DataFrame(state["history"])
    start_price = float(history["price"].iloc[0])
    return {
        "opened_on": state["opened_on"], "last_mark": broker.last_date, "instrument": state["instrument"],
        "equity": round(equity, 2), "cash": round(broker.cash, 2),
        "position_value": round(broker.units * (broker.last_price or 0.0), 2),
        "weight": round(broker.weight(), 4),
        "return_pct": round((equity / state["initial_cash"] - 1.0) * 100.0, 3),
        "index_return_pct": round(((broker.last_price or start_price) / start_price - 1.0) * 100.0, 3),
        "trades": len(broker.trades), "costs_paid": round(broker.costs_paid, 2),
        "interest_earned": round(broker.interest_earned, 2),
        "days": len(history) - 1, "pending": state.get("pending"),
    }
