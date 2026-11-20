"""Data files for the decision dashboard (reports/dashboard/*.json, each {rows, meta}).

The dashboard page draws only from these files, so every number on it can be traced to one of
them, and the daily task refreshes them by re-running this module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Paths
from .features.panel import PanelBundle
from .stocks import combine, history, setups
from .trading import paper

log = logging.getLogger(__name__)

SETUP_HE = {"oversold": "מכירת יתר (RSI מתחת ל-30)", "oversold_bounce": "התאוששות ממכירת יתר",
            "breakout_20d": "פריצה לשיא 20 יום", "bull_flag": "דגל עלייה", "reclaim_ma20": "חזרה מעל ממוצע 20 יום"}
REGIME_HE = {"pessimistic": "פסימי", "neutral": "ניטרלי", "optimistic": "אופטימי"}
STRATEGY_HE = {"cnn_all": "המודל (CNN)", "buy_and_hold": "קנה והחזק", "vol_target_15": "יעד תנודתיות 15%",
               "trend_200d": "מגמה 200 יום", "cnn_breadth": "CNN + רוחב שוק", "constant_mix": "חשיפה קבועה זהה"}


def out_dir(paths: Paths) -> Path:
    return paths.reports / "dashboard"


def _clean(value):
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 6)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value.date())
    return value


def _write(folder: Path, name: str, rows: list[dict], meta: list[dict]) -> Path:
    rows = [{k: _clean(v) for k, v in r.items()} for r in rows]
    path = folder / f"{name}.json"
    path.write_text(json.dumps({"rows": rows, "meta": meta}, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return path


def build(cfg: dict, paths: Paths, bundle: PanelBundle, pred: pd.DataFrame) -> dict[str, Path]:
    folder = out_dir(paths)
    folder.mkdir(parents=True, exist_ok=True)
    horizon = int(cfg["model"]["trade_horizon"])
    written: dict[str, Path] = {}
    origin = [{"label": "Made by", "value": "python -m marketlab dashboard (marketlab/dashboard.py)"}]

    # ------------------------------------------------------------------ today's decision
    signal_path = paths.state / "latest_signal.json"
    if signal_path.exists():
        s = json.loads(signal_path.read_text(encoding="utf-8"))
        acc = s["account"]
        row = {"date": s["date"], "decision": s["decision"], "decision_he": s["decision_he"],
               "weight_now": s["weight_now"], "weight_target": s["weight_target"], "p_up": s["p_up"],
               "p_usual": s.get("p_usual"), "score": s.get("score"), "horizon_days": s["horizon_days"],
               "equity": acc["equity"], "return_pct": acc["return_pct"], "index_return_pct": acc["index_return_pct"]}
        written["signal"] = _write(folder, "signal", [row], origin + [{"label": "Input", "value": "state/latest_signal.json"}])

    # ------------------------------------------------------------------ stocks: breadth, today's setups, track record
    closes, _ = history.load_history(paths)
    members = history.load_members(paths)
    state = None
    if closes is not None:
        flags = setups.setup_flags(closes)
        br = setups.breadth(closes, flags)
        weekly = br.dropna(subset=["br_above_ma200"]).iloc[::-5].iloc[::-1]  # every 5th day, ending today
        written["breadth"] = _write(folder, "breadth", [
            {"date": d, "above_ma200": r.br_above_ma200, "above_ma50": r.br_above_ma50, "oversold": r.br_oversold,
             "breakout_20d": r.br_breakout_20d, "members": r.br_members} for d, r in weekly.iterrows()],
            origin + [{"label": "What", "value": "Share of NASDAQ-100 members (today's list) in each state, every 5th trading day"}])

        last = closes.index[-1]
        names = members.set_index("symbol") if members is not None else pd.DataFrame()
        rsi = flags["_rsi"]
        today_rows = []
        for sym in closes.columns:
            tags = [k for k in setups.SETUPS if flags[k].at[last, sym]]
            for tag in tags:
                today_rows.append({"symbol": sym, "company": names["company"].get(sym, "") if len(names) else "",
                                   "industry": names["industry"].get(sym, "") if len(names) else "",
                                   "setup": tag, "setup_he": SETUP_HE[tag], "close": float(closes.at[last, sym]),
                                   "rsi": float(rsi.at[last, sym]), "date": last})
        written["setups_today"] = _write(folder, "setups_today", today_rows, origin + [
            {"label": "Rules", "value": "trader-scanner find_setups, on daily closes of NASDAQ-100 members"}])

        dev_end = int(cfg["walkforward"]["dev_end_year"])
        periods = {"development": (str(pred.index[0].date()), f"{dev_end}-12-31"),
                   "holdout": (f"{dev_end + 1}-01-01", str(pred.index[-1].date()))}
        study = combine.setup_regime_study(closes, flags, bundle.price, pred, horizon, periods, horizons=(20,))
        study = study[study["regime"] != "all"]
        written["setup_record"] = _write(folder, "setup_record", [
            {"period": r.period, "setup": r.setup, "setup_he": SETUP_HE.get(r.setup, "כל המניות"),
             "regime": r.regime, "regime_he": REGIME_HE[r.regime], "events": r.events,
             "mean_excess_20d": r.mean_excess, "hit_excess_20d": r.hit_excess, "t": r.t_excess}
            for r in study.itertuples()], origin + [
            {"label": "What", "value": "Stock return over the next 20 trading days minus the index's, by setup and by what the CNN said that day"},
            {"label": "Caveat", "value": "Members are today's NASDAQ-100 (survivors), which flatters past stock returns"}])

        state = combine.analog_frame(bundle, br, pred, horizon)

    # ------------------------------------------------------------------ outlook from similar past days
    if state is not None:
        outlook = combine.analog_outlook(state, state.index[-1])
        check = combine.validate_analogs(state, "2014-01-01")
        if outlook:
            rows = []
            for h in (5, 20):
                o = outlook[f"h{h}"]
                rows.append({"horizon_days": h, "p_up": o["p_up"], "base_p_up": o["base_p_up"],
                             **{f"q{int(q * 100)}": v for q, v in zip([0.1, 0.25, 0.5, 0.75, 0.9], o["quantiles"])},
                             **{f"base_q{int(q * 100)}": v for q, v in zip([0.1, 0.25, 0.5, 0.75, 0.9], o["base_quantiles"])},
                             "similar_days": outlook["k"], "pool_days": outlook["pool_days"], "date": outlook["date"]})
            written["outlook"] = _write(folder, "outlook", rows, origin + [
                {"label": "How", "value": "The 60 past days (one per 10-day episode) closest to today in: " + ", ".join(outlook["features"])},
                {"label": "Walk-forward check", "value": f"IC20 {check.get('ic20', float('nan')):.3f}, "
                                                       f"80% band held {check.get('band_80_coverage', float('nan')):.0%} of outcomes"}])
            written["similar_days"] = _write(folder, "similar_days", outlook["similar_days"], origin)
            written["outlook_check"] = _write(folder, "outlook_check", [{
                "ic20": check.get("ic20"), "ic5": check.get("ic5"), "band_80_coverage": check.get("band_80_coverage"),
                "days": check.get("days"), "hit20_when_p_above_base": check.get("hit20_when_p_above_base"),
                "hit20_when_p_below_base": check.get("hit20_when_p_below_base")}], origin)

    # ------------------------------------------------------------------ how far to trust the model
    strategies_path = paths.runs / "latest" / "strategies.csv"
    if strategies_path.exists():
        st = pd.read_csv(strategies_path)
        keep = st[st["strategy"].isin(STRATEGY_HE) & ~((st["variant"] == "rules") & (st["strategy"] == "trend_200d"))]
        keep = keep.drop_duplicates(["period", "strategy"])
        written["scorecard"] = _write(folder, "scorecard", [
            {"period": r.period, "strategy": r.strategy, "strategy_he": STRATEGY_HE[r.strategy], "cagr": r.cagr,
             "sharpe": r.sharpe, "max_drawdown": r.max_drawdown, "avg_exposure": r.avg_exposure,
             "timing_pvalue": r.timing_pvalue} for r in keep.itertuples() if r.period in ("development", "holdout")],
            origin + [{"label": "Input", "value": "runs/latest/strategies.csv (walk-forward backtest)"}])
    equity_path = paths.runs / "latest" / "equity_full.csv"
    if equity_path.exists():
        eq = pd.read_csv(equity_path, index_col=0, parse_dates=True)
        weekly = eq.iloc[::-5].iloc[::-1]
        written["track"] = _write(folder, "track", [
            {"date": d, "model": r["cnn_all"] / eq["cnn_all"].iloc[0] * 100, "buy_and_hold": r["buy_and_hold"] / eq["buy_and_hold"].iloc[0] * 100,
             "exposure": r.get("cnn_all_weight")} for d, r in weekly.iterrows()],
            origin + [{"label": "What", "value": "Value of 100 in the backtest, every 5th trading day"}])

    # ------------------------------------------------------------------ virtual account
    account = paper.load_state(paths.state)
    if account is not None:
        written["paper"] = _write(folder, "paper", [
            {"date": h["date"], "price": h["price"], "equity": h["equity"], "weight": h["weight"], "traded": bool(h.get("traded"))}
            for h in account["history"]], origin + [{"label": "Input", "value": "state/paper_account.json"}])
    log.info("dashboard data: %s", ", ".join(sorted(written)))
    return written
