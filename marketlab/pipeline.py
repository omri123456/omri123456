"""The steps of the system, each callable on its own: the CLI is a thin wrapper around these.

fetch -> research -> train (walk-forward) -> backtest -> paper account -> report
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Paths
from .data import store
from .features.panel import EXTRA_GROUPS, GROUPS, PanelBundle, build_panel
from .model import train
from .model.dataset import make_labels
from .research import event_study, feature_ic
from .trading import evaluate, metrics, paper

log = logging.getLogger(__name__)

# Model variants compared in every run. MAIN is the one the paper account trades.
VARIANTS = {
    "cnn_all": {"kind": "cnn", "groups": GROUPS, "title": "CNN, all information"},
    "cnn_rules": {"kind": "cnn", "groups": GROUPS + EXTRA_GROUPS, "title": "CNN, all information + classic rule signals"},
    "cnn_market": {"kind": "cnn", "groups": ["market", "calendar"], "title": "CNN, prices only"},
    "linear_all": {"kind": "linear", "groups": GROUPS, "title": "Linear baseline, all information"},
}
MAIN = "cnn_all"


def run_dir(paths: Paths) -> Path:
    path = paths.runs / "latest"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_bundle(cfg: dict, paths: Paths) -> tuple[store.RawData, PanelBundle]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # openpyxl complains about spreadsheet headers
        raw = store.load_raw(cfg, paths)
    return raw, build_panel(raw, cfg)


# ---------------------------------------------------------------------------------------------
def do_fetch(cfg: dict, paths: Paths) -> dict:
    summary = store.fetch_everything(cfg, paths)
    log.info("fetch: %s", json.dumps(summary, default=str))
    return summary


def do_build(cfg: dict, paths: Paths) -> PanelBundle:
    """Write the panel to disk for inspection (the other steps rebuild it in memory)."""
    _, bundle = load_bundle(cfg, paths)
    paths.processed.mkdir(parents=True, exist_ok=True)
    bundle.panel.to_csv(paths.processed / "panel.csv")
    pd.DataFrame({"price": bundle.price, "cash_rate": bundle.cash_rate}).to_csv(paths.processed / "prices.csv")
    with open(paths.processed / "groups.json", "w", encoding="utf-8") as fh:
        json.dump(bundle.groups, fh, indent=1)
    log.info("panel: %d days x %d features (%s to %s)", *bundle.panel.shape,
             bundle.panel.index[0].date(), bundle.panel.index[-1].date())
    return bundle


def do_research(cfg: dict, paths: Paths) -> dict:
    raw, bundle = load_bundle(cfg, paths)
    out = paths.runs / "research"
    out.mkdir(parents=True, exist_ok=True)
    start = cfg["project"]["start"]
    summary, path_table = event_study.study(raw, bundle, start=start)
    summary.to_csv(out / "event_study.csv")
    path_table.to_csv(out / "event_paths.csv")
    ic = feature_ic.feature_ic(bundle, cfg, int(cfg["walkforward"]["dev_end_year"]))
    ic.to_csv(out / "feature_ic.csv", index=False)
    baseline = event_study.baseline_hit_rates(bundle, start)
    with open(out / "baseline.json", "w", encoding="utf-8") as fh:
        json.dump(baseline, fh, indent=1)
    log.info("research: %d event types, %d feature/horizon pairs", len(summary), len(ic))
    return {"events": summary, "paths": path_table, "ic": ic, "baseline": baseline}


# ---------------------------------------------------------------------------------------------
def pred_path(paths: Paths, name: str) -> Path:
    return run_dir(paths) / f"pred_{name}.csv"


def read_pred(paths: Paths, name: str) -> pd.DataFrame | None:
    path = pred_path(paths, name)
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


def attach_labels(pred: pd.DataFrame, bundle: PanelBundle, cfg: dict) -> pd.DataFrame:
    """(Re)compute realised returns for every prediction row from the current price history."""
    horizons = [int(h) for h in cfg["model"]["horizons"]]
    labels = make_labels(bundle.price, horizons, int(cfg["model"]["execution_lag"]))
    pos = bundle.price.index.get_indexer(pred.index)
    pred = pred[pos >= 0].copy()
    pos = pos[pos >= 0]
    for j, h in enumerate(horizons):
        pred[f"ret_{h}"] = labels.ret[pos, j]
        pred[f"up_{h}"] = labels.up[pos, j]
    return pred


def do_train(cfg: dict, paths: Paths, variants: list[str] | None = None,
             years: list[int] | None = None) -> None:
    _, bundle = load_bundle(cfg, paths)
    for name in variants or list(VARIANTS):
        spec = VARIANTS[name]
        save_last = paths.models / "production.pt" if name == MAIN else None
        if years and save_last is not None and max(years) < bundle.panel.index[-1].year:
            save_last = None  # only the most recent fold may become the production model
        pred = train.walk_forward(bundle, cfg, kind=spec["kind"], groups=spec["groups"], years=years,
                                  save_last=save_last)
        existing = read_pred(paths, name)
        if years and existing is not None:  # partial retrain: replace only the retrained folds
            keep = existing[~existing["fold"].isin(pred["fold"].unique())]
            pred = pd.concat([keep, pred]).sort_index()
        pred.to_csv(pred_path(paths, name))
        log.info("train: %s -> %d out-of-sample days (%s to %s)", name, len(pred),
                 pred.index[0].date(), pred.index[-1].date())


# ---------------------------------------------------------------------------------------------
def _periods(cfg: dict, pred: pd.DataFrame) -> dict[str, tuple[str, str]]:
    dev_end = int(cfg["walkforward"]["dev_end_year"])
    first, last = pred.index[0], pred.index[-1]
    return {
        "development": (str(first.date()), f"{dev_end}-12-31"),
        "holdout": (f"{dev_end + 1}-01-01", str(last.date())),
        "full": (str(first.date()), str(last.date())),
    }


def group_ablation(pred: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Out-of-sample effect of neutralising each feature group (set to its training mean)."""
    rows = []
    base_cols = [f"p_{horizon}", f"up_{horizon}", f"ret_{horizon}"]
    for column in pred.columns:
        prefix = f"p_{horizon}_wo_"
        if not column.startswith(prefix):
            continue
        frame = pred[base_cols + [column]].dropna()
        if len(frame) < 100:
            continue
        up, ret = frame[f"up_{horizon}"].to_numpy(), frame[f"ret_{horizon}"]
        ll_full = metrics.log_loss(frame[f"p_{horizon}"].to_numpy(), up)
        ll_without = metrics.log_loss(frame[column].to_numpy(), up)
        rows.append({
            "group": column[len(prefix):],
            # positive = the model predicts better WITH the group than without it
            "logloss_gain": ll_without - ll_full,
            "ic_full": metrics.spearman(frame[f"p_{horizon}"], ret),
            "ic_without": metrics.spearman(frame[column], ret),
        })
    return pd.DataFrame(rows).set_index("group") if rows else pd.DataFrame()


def do_backtest(cfg: dict, paths: Paths) -> dict:
    """Score every variant out of sample and simulate the strategies with virtual money."""
    raw, bundle = load_bundle(cfg, paths)
    out = run_dir(paths)
    horizons = [int(h) for h in cfg["model"]["horizons"]]
    trade_h = int(cfg["model"]["trade_horizon"])
    results: dict = {"quality": [], "strategies": [], "yearly": {}, "ablation": {}}
    curves_main: dict[str, dict[str, pd.DataFrame]] = {}

    for name in VARIANTS:
        pred = read_pred(paths, name)
        if pred is None:
            continue
        pred = attach_labels(pred, bundle, cfg)
        for period, (start, end) in _periods(cfg, pred).items():
            window = pred.loc[start:end]
            if len(window) < 60:
                continue
            for h in horizons:
                quality = metrics.prediction_quality(window, h)
                if quality:
                    results["quality"].append({"variant": name, "period": period, "horizon": h, **quality})
            table, curves, _ = evaluate.compare(pred, bundle, cfg, label=name, start=start, end=end)
            for strategy, row in table.iterrows():
                if strategy != name and name != MAIN:
                    continue  # benchmarks are identical across variants: keep one copy
                results["strategies"].append({"variant": name, "period": period, "strategy": strategy,
                                              **row.to_dict()})
            if name == MAIN:
                curves_main[period] = curves
                results["yearly"][period] = evaluate.yearly_returns(curves)
        if name == MAIN:
            for period, (start, end) in _periods(cfg, pred).items():
                results["ablation"][period] = group_ablation(pred.loc[start:end], trade_h)

    main_pred = read_pred(paths, MAIN)
    if main_pred is not None:  # classic rule-based methods over the same periods
        for period, (start, end) in _periods(cfg, main_pred).items():
            table, _ = evaluate.classic_rules(bundle, cfg, start=start, end=end)
            for strategy, row in table.iterrows():
                if len(main_pred.loc[start:end]) >= 60:
                    results["strategies"].append({"variant": "rules", "period": period, "strategy": strategy,
                                                  **row.to_dict()})

    quality = pd.DataFrame(results["quality"])
    strategies = pd.DataFrame(results["strategies"])
    quality.to_csv(out / "prediction_quality.csv", index=False)
    strategies.to_csv(out / "strategies.csv", index=False)
    for period, curves in curves_main.items():
        frame = pd.DataFrame({name: eq["equity"] for name, eq in curves.items()})
        frame[f"{MAIN}_weight"] = curves[MAIN]["weight"]
        frame.to_csv(out / f"equity_{period}.csv")
        results["yearly"][period].to_csv(out / f"yearly_{period}.csv")
        if len(results["ablation"].get(period, [])):
            results["ablation"][period].to_csv(out / f"ablation_{period}.csv")
    results["quality"], results["strategies"], results["curves"] = quality, strategies, curves_main
    main = read_pred(paths, MAIN)
    info = {
        "data_last_day": str(bundle.panel.index[-1].date()),
        "panel_days": int(bundle.panel.shape[0]), "panel_features": int(bundle.panel.shape[1]),
        "events": int(len(raw.events)) if raw.events is not None else 0,
        "seeds": len(cfg["model"]["seeds"]),
        "first_test_year": int(main["fold"].min()) if main is not None else None,
        "last_test_year": int(main["fold"].max()) if main is not None else None,
    }
    model_path = paths.models / "production.pt"
    if model_path.exists():
        info["parameters"] = int(train.load_production(model_path)["parameters"])
    with open(out / "run_info.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    log.info("backtest: %d quality rows, %d strategy rows", len(quality), len(strategies))
    return results


# ---------------------------------------------------------------------------------------------
def extend_predictions(cfg: dict, paths: Paths, bundle: PanelBundle) -> pd.DataFrame:
    """Predict any trading day newer than the stored predictions with the production model."""
    pred = read_pred(paths, MAIN)
    if pred is None:
        raise RuntimeError("no predictions yet: run the train step first")
    model_path = paths.models / "production.pt"
    newest = bundle.panel.index[-1]
    if newest <= pred.index[-1]:
        return pred
    if not model_path.exists():
        raise RuntimeError("production model missing: run the train step first")
    production = train.load_production(model_path)
    if newest.year > int(production["fold"]):
        log.warning("the production model was trained for %s; a new year started, so retrain "
                    "(python -m marketlab train) to follow the yearly schedule", production["fold"])
    fresh = train.predict_production(production, bundle, after=pred.index[-1])
    pred = pd.concat([pred, fresh])
    pred.to_csv(pred_path(paths, MAIN))
    log.info("predicted %d new day(s) up to %s", len(fresh), newest.date())
    return pred


def do_paper(cfg: dict, paths: Paths, action: str = "step", start: str | None = None,
             force: bool = False) -> dict:
    _, bundle = load_bundle(cfg, paths)
    pred = extend_predictions(cfg, paths, bundle)
    state = paper.load_state(paths.state)
    if action == "init" or state is None:
        if state is not None and not force:
            raise RuntimeError("a paper account already exists; pass --force to start over")
        state = paper.init_account(bundle, pred, cfg, paths.state, start)
    elif action == "step":
        state = paper.advance(bundle, pred, cfg, state)
        paper.save_state(paths.state, state)
    info = paper.summary(state)
    log.info("paper account: %s", json.dumps(info, default=str))
    return info


def do_status(cfg: dict, paths: Paths) -> dict:
    raw, bundle = load_bundle(cfg, paths)
    fresh = store.freshness(raw)
    info = {
        "data_last_day": str(bundle.panel.index[-1].date()),
        "panel_shape": list(bundle.panel.shape),
        "oldest_source": str(fresh["last_obs"].min().date()) if len(fresh) else None,
        "events": int(len(raw.events)) if raw.events is not None else 0,
    }
    pred = read_pred(paths, MAIN)
    if pred is not None:
        info["predictions_through"] = str(pred.index[-1].date())
    model_path = paths.models / "production.pt"
    if model_path.exists():
        production = train.load_production(model_path)
        info["model"] = {k: production[k] for k in ("kind", "fold", "trained_through", "parameters")}
    state = paper.load_state(paths.state)
    if state is not None:
        info["paper_account"] = paper.summary(state)
    return info


def as_jsonable(obj):
    """Make numpy / pandas scalars printable as JSON."""
    if isinstance(obj, dict):
        return {str(k): as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj.date())
    return obj
