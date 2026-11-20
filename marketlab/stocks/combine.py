"""Joining the two systems.

1. setup_regime_study: the day-trader scanner's stock setups, scored by what the stock did next,
   split by what the CNN said about the whole index that day. If the setups work better when
   the CNN is optimistic, the two systems are worth combining; if not, the CNN adds nothing
   to stock picking.
2. analog_outlook: for a given day, the past days whose state looked most alike (the CNN's
   view, volatility, drawdown, momentum, breadth, news tone) and what the index did after them.
   It is validated walk-forward: every day is forecast only from days already finished.

Everything uses the CNN's out-of-sample predictions (2012 on), so the CNN never judges a day it
was trained on. Stock members are today's NASDAQ-100: survivors, so stock returns in the past are
flattered. Comparisons inside the same set of stocks (regime A vs regime B) are fairer than levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..model.dataset import make_labels
from ..trading.metrics import spearman
from .setups import SETUPS

REGIMES = {"pessimistic": "המודל פסימי", "neutral": "המודל ניטרלי", "optimistic": "המודל אופטימי"}


def cnn_score(pred: pd.DataFrame, horizon: int) -> pd.Series:
    spread = pred[f"ref_std_{horizon}"].replace(0.0, np.nan)
    return ((pred[f"p_{horizon}"] - pred[f"ref_mean_{horizon}"]) / spread).rename("score")


def regime_of(score: pd.Series, cut: float = 0.5) -> pd.Series:
    out = pd.Series("neutral", index=score.index)
    out[score < -cut] = "pessimistic"
    out[score > cut] = "optimistic"
    return out.where(score.notna())


def _forward(log_prices: pd.DataFrame | pd.Series, h: int, lag: int = 1):
    """Log return from the close `lag` days after t to `lag + h` days after t (the fill we can get)."""
    return log_prices.shift(-(lag + h)) - log_prices.shift(-lag)


def _clustered_t(values: np.ndarray, blocks: np.ndarray) -> float:
    """Mean / standard error with errors clustered by block (same day and overlapping horizons)."""
    if len(values) < 30:
        return np.nan
    dev = values - values.mean()
    sums = pd.Series(dev).groupby(blocks).sum().to_numpy()
    se = np.sqrt((sums ** 2).sum()) / len(values)
    return float(values.mean() / se) if se > 0 else np.nan


def setup_regime_study(closes: pd.DataFrame, flags: dict[str, pd.DataFrame], index_price: pd.Series,
                       pred: pd.DataFrame, horizon: int, periods: dict[str, tuple[str, str]],
                       horizons: tuple[int, ...] = (5, 20)) -> pd.DataFrame:
    """One row per (period, setup, regime, horizon): events, mean raw and excess return, hit rate, t."""
    logc = np.log(closes.astype(float)).ffill(limit=3)
    logi = np.log(index_price.astype(float)).reindex(closes.index).ffill(limit=3)
    regime = regime_of(cnn_score(pred, horizon)).reindex(closes.index)
    day_pos = pd.Series(np.arange(len(closes.index)), index=closes.index)
    alive = closes.notna() & (closes.rolling(30, min_periods=30).count() >= 30)
    rows = []
    for h in horizons:
        fwd = _forward(logc, h)
        excess = fwd.sub(_forward(logi, h), axis=0)
        for setup in ["all", *SETUPS]:
            mask = alive if setup == "all" else (flags[setup] & alive)
            stacked = pd.DataFrame({
                "raw": fwd.where(mask).stack(),
                "excess": excess.where(mask).stack(),
            }).dropna()
            if stacked.empty:
                continue
            dates = stacked.index.get_level_values(0)
            stacked["regime"] = regime.reindex(dates).to_numpy()
            stacked["block"] = (day_pos.reindex(dates).to_numpy() // h)
            for period, (start, end) in periods.items():
                part = stacked[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
                for reg in ["all", *REGIMES]:
                    sub = part if reg == "all" else part[part["regime"] == reg]
                    sub = sub.dropna(subset=["regime"]) if reg == "all" else sub
                    if len(sub) < 30:
                        continue
                    rows.append({
                        "period": period, "setup": setup, "regime": reg, "horizon": h,
                        "events": len(sub), "days": int(sub.index.get_level_values(0).nunique()),
                        "mean_raw": float(np.expm1(sub["raw"]).mean()),
                        "mean_excess": float(sub["excess"].mean()),
                        "hit_raw": float((sub["raw"] > 0).mean()),
                        "hit_excess": float((sub["excess"] > 0).mean()),
                        "t_excess": _clustered_t(sub["excess"].to_numpy(), sub["block"].to_numpy()),
                        "t_raw": _clustered_t(sub["raw"].to_numpy(), sub["block"].to_numpy()),
                    })
    return pd.DataFrame(rows)


def regime_gap(study: pd.DataFrame, period: str, horizon: int) -> pd.DataFrame:
    """Per setup: return when the CNN is optimistic minus when it is pessimistic."""
    part = study[(study["period"] == period) & (study["horizon"] == horizon)]
    out = []
    for setup, g in part.groupby("setup"):
        g = g.set_index("regime")
        if not {"optimistic", "pessimistic"} <= set(g.index):
            continue
        out.append({"setup": setup,
                    "raw_optimistic": g.at["optimistic", "mean_raw"], "raw_pessimistic": g.at["pessimistic", "mean_raw"],
                    "gap_raw": g.at["optimistic", "mean_raw"] - g.at["pessimistic", "mean_raw"],
                    "excess_optimistic": g.at["optimistic", "mean_excess"],
                    "excess_pessimistic": g.at["pessimistic", "mean_excess"],
                    "events_optimistic": int(g.at["optimistic", "events"]),
                    "events_pessimistic": int(g.at["pessimistic", "events"])})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------
# Analog outlook
# ---------------------------------------------------------------------------------------------

ANALOG_FEATURES = ["score", "vix_z", "drawdown", "ret_20d", "br_above_ma200", "br_oversold", "news_sent"]


def analog_frame(bundle, breadth: pd.DataFrame, pred: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """State of each day (only days with an out-of-sample CNN view) plus the index's future."""
    state = pd.DataFrame(index=pred.index)
    state["score"] = cnn_score(pred, horizon)
    for col in ("vix_z", "drawdown", "ret_20d", "news_sent"):
        if col in bundle.panel.columns:
            state[col] = bundle.panel[col].reindex(state.index)
    for col in ("br_above_ma200", "br_oversold"):
        state[col] = breadth[col].reindex(state.index)
    labels = make_labels(bundle.price, [5, 20], 1)
    fwd = pd.DataFrame(labels.ret, index=bundle.price.index, columns=["fwd5", "fwd20"])
    state = state.join(fwd, how="left")
    state["pos"] = bundle.price.index.get_indexer(state.index)
    return state


def _neighbours(pool: pd.DataFrame, target: pd.Series, cols: list[str], k: int, spacing: int) -> pd.DataFrame:
    x = pool[cols].to_numpy(float)
    mu, sd = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
    sd[sd == 0] = 1.0
    z = (x - mu) / sd
    t = (target[cols].to_numpy(float) - mu) / sd
    usable = ~np.isnan(t)
    d = np.sqrt(np.nansum((z[:, usable] - t[usable]) ** 2, axis=1) / max(usable.sum(), 1))
    order = np.argsort(d)
    chosen, taken = [], []
    positions = pool["pos"].to_numpy()
    for i in order:
        if np.isnan(d[i]):
            continue
        if all(abs(positions[i] - p) >= spacing for p in taken):  # one per episode, not 5 copies of a week
            chosen.append(i)
            taken.append(positions[i])
            if len(chosen) == k:
                break
    out = pool.iloc[chosen].copy()
    out["distance"] = d[chosen]
    return out


def analog_outlook(state: pd.DataFrame, day: pd.Timestamp, k: int = 60, spacing: int = 10) -> dict | None:
    """What the index did after the k past days that looked most like `day` (labels known by then)."""
    if day not in state.index:
        return None
    target = state.loc[day]
    cols = [c for c in ANALOG_FEATURES if c in state.columns and pd.notna(target.get(c))]
    pool = state[(state["pos"] + 1 + 20 < target["pos"]) & state["fwd20"].notna()].dropna(subset=cols)
    if len(pool) < 3 * k or len(cols) < 3:
        return None
    near = _neighbours(pool, target, cols, k, spacing)
    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    result = {"date": str(day.date()), "k": len(near), "features": cols, "pool_days": len(pool)}
    for h in (5, 20):
        r = np.expm1(near[f"fwd{h}"].to_numpy())
        base = np.expm1(pool[f"fwd{h}"].to_numpy())
        result[f"h{h}"] = {"p_up": float((r > 0).mean()), "quantiles": [float(np.quantile(r, x)) for x in q],
                           "mean": float(r.mean()), "base_p_up": float((base > 0).mean()),
                           "base_quantiles": [float(np.quantile(base, x)) for x in q]}
    nearest = near.nsmallest(8, "distance")
    result["similar_days"] = [{"date": str(d.date()), "fwd5": float(np.expm1(r["fwd5"])), "fwd20": float(np.expm1(r["fwd20"])),
                               "score": float(r["score"]), "vix_z": float(r.get("vix_z", np.nan)),
                               "drawdown": float(r.get("drawdown", np.nan))}
                              for d, r in nearest.iterrows()]
    result["today"] = {c: float(target[c]) for c in cols}
    return result


def validate_analogs(state: pd.DataFrame, start: str, k: int = 60, spacing: int = 10, every: int = 5) -> dict:
    """Walk-forward check: forecast every `every`-th day from earlier days only, compare with what happened."""
    days = state.index[(state.index >= pd.Timestamp(start)) & state["fwd20"].notna()][::every]
    rows = []
    for day in days:
        out = analog_outlook(state, day, k, spacing)
        if out is None:
            continue
        rows.append({"date": day, "pred_p20": out["h20"]["p_up"], "pred_med20": out["h20"]["quantiles"][2],
                     "pred_p5": out["h5"]["p_up"], "base_p20": out["h20"]["base_p_up"],
                     "real20": float(np.expm1(state.at[day, "fwd20"])), "real5": float(np.expm1(state.at[day, "fwd5"])),
                     "q10": out["h20"]["quantiles"][0], "q90": out["h20"]["quantiles"][4]})
    frame = pd.DataFrame(rows).set_index("date")
    if frame.empty:
        return {}
    inside = ((frame["real20"] >= frame["q10"]) & (frame["real20"] <= frame["q90"])).mean()
    return {
        "days": len(frame), "start": str(frame.index[0].date()), "end": str(frame.index[-1].date()),
        "ic20": spearman(frame["pred_p20"], frame["real20"]),
        "ic5": spearman(frame["pred_p5"], frame["real5"]),
        "hit20_when_p_above_base": float((frame.loc[frame["pred_p20"] > frame["base_p20"], "real20"] > 0).mean()),
        "hit20_when_p_below_base": float((frame.loc[frame["pred_p20"] <= frame["base_p20"], "real20"] > 0).mean()),
        "share_p_above_base": float((frame["pred_p20"] > frame["base_p20"]).mean()),
        "band_80_coverage": float(inside),
        "series": frame,
    }
