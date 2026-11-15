"""Training, walk-forward evaluation and the production ensemble.

Walk-forward: for every test year the model is trained from scratch on data whose labels were
fully known before that year started, then asked to predict the year day by day. Nothing from
the future (data, scaling statistics, labels) reaches the model.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..features.panel import PanelBundle
from .cnn import LinearBaseline, MarketCNN, count_parameters
from .dataset import Labels, Scaler, make_labels, make_windows, train_rows_before

log = logging.getLogger(__name__)

# Which feature groups share a CNN branch.
BRANCHES = {
    "market": ["market", "rules"],
    "rates": ["rates", "cross", "liquidity"],
    "macro": ["macro", "forecasts"],
    "news": ["news_idx", "wiki"],
    "calendar": ["calendar"],
}


@dataclass
class Design:
    """The columns a model variant uses and how they map onto branches."""

    columns: list[str]
    branch_index: dict[str, list[int]]

    @classmethod
    def from_bundle(cls, bundle: PanelBundle, groups: list[str] | None = None) -> "Design":
        wanted = set(groups) if groups else set(bundle.groups.values())
        columns = [c for c in bundle.panel.columns if bundle.groups[c] in wanted]
        branch_index = {
            name: [i for i, c in enumerate(columns) if bundle.groups[c] in members]
            for name, members in BRANCHES.items()
        }
        return cls(columns=columns, branch_index=branch_index)


def build_model(kind: str, design: Design, n_horizons: int, mcfg: dict) -> nn.Module:
    if kind == "linear":
        return LinearBaseline(len(design.columns), n_horizons)
    return MarketCNN(design.branch_index, n_horizons, hidden=int(mcfg["hidden"]),
                     dropout=float(mcfg["dropout"]))


def train_one(model: nn.Module, x: torch.Tensor, up: torch.Tensor, z: torch.Tensor,
              mcfg: dict, seed: int) -> nn.Module:
    """Fixed-length training run (no early stopping: nothing is tuned on later data)."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    epochs, batch = int(mcfg["epochs"]), int(mcfg["batch_size"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(mcfg["lr"]),
                                  weight_decay=float(mcfg["weight_decay"]))
    steps = epochs * max(1, (len(x) + batch - 1) // batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=float(mcfg["lr"]),
                                                    total_steps=steps, pct_start=0.2)
    bce = nn.BCEWithLogitsLoss()
    huber = nn.HuberLoss(delta=1.5)
    reg_weight = float(mcfg["reg_weight"])
    z = z.clamp(-5.0, 5.0)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), batch):
            idx = order[start:start + batch]
            if len(idx) < 2:  # batch norm needs more than one sample
                continue
            logits, magnitude = model(x[idx])
            loss = bce(logits, up[idx]) + reg_weight * huber(magnitude, z[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    return model.eval()


@torch.no_grad()
def predict(models: list[nn.Module], x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble average of P(up) and of the expected scaled return: both [N, H]."""
    if len(x) == 0:
        n_h = 0
        return np.zeros((0, n_h)), np.zeros((0, n_h))
    probs, mags = [], []
    for model in models:
        model.eval()
        logits, magnitude = model(x)
        probs.append(torch.sigmoid(logits).numpy())
        mags.append(magnitude.numpy())
    return np.mean(probs, axis=0), np.mean(mags, axis=0)


def predict_with_reference(models: list[nn.Module], scaled: np.ndarray, rows: np.ndarray, window: int,
                           reference: int, horizons: list[int]) -> pd.DataFrame:
    """Predictions for `rows` plus, for each, the mean and spread of the same ensemble's
    predictions over the `reference` days before it.

    The reference lets us read a prediction as "more or less optimistic than this model has
    been lately", which stays meaningful even when the model's absolute level is off in a new
    regime. `rows` must be contiguous; earlier rows are predicted only to fill the reference.
    """
    first = int(rows[0])
    warm_start = max(window - 1, first - reference)
    all_rows = np.arange(warm_start, int(rows[-1]) + 1)
    prob, magnitude = predict(models, make_windows(scaled, all_rows, window))
    skip = len(all_rows) - len(rows)
    out = {}
    for j, h in enumerate(horizons):
        history = pd.Series(prob[:, j]).shift(1)  # strictly before the day being scored
        out[f"p_{h}"] = prob[skip:, j]
        out[f"mu_{h}"] = magnitude[skip:, j]
        out[f"ref_mean_{h}"] = history.rolling(reference, min_periods=40).mean().to_numpy()[skip:]
        out[f"ref_std_{h}"] = history.rolling(reference, min_periods=40).std().to_numpy()[skip:]
    return pd.DataFrame(out)


def _fit_ensemble(kind: str, design: Design, matrix: np.ndarray, labels: Labels, rows: np.ndarray,
                  mcfg: dict) -> tuple[list[nn.Module], Scaler, np.ndarray]:
    """Train one model per seed on the given rows. Returns models, scaler and the scaled matrix."""
    window = int(mcfg["window"])
    scaler = Scaler.fit(matrix[: rows.max() + 1], float(mcfg["clip_sigma"]))
    scaled = scaler.transform(matrix)
    x = make_windows(scaled, rows, window)
    up = torch.from_numpy(labels.up[rows].astype(np.float32))
    z = torch.from_numpy(labels.z[rows].astype(np.float32))
    models = []
    for seed in mcfg["seeds"]:
        model = build_model(kind, design, len(labels.horizons), mcfg)
        with torch.no_grad():  # start from the base rate instead of 50/50
            base = up.mean(dim=0).clamp(0.05, 0.95)
            model.direction.bias.copy_(torch.log(base / (1 - base)))
            model.magnitude.bias.copy_(z.clamp(-5, 5).mean(dim=0))
        models.append(train_one(model, x, up, z, mcfg, int(seed)))
    return models, scaler, scaled


def _init_worker() -> None:
    torch.set_num_threads(1)  # one fold per core beats several threads on one small model


def _run_fold(job: dict) -> pd.DataFrame | None:
    """Train on everything known before `year` and predict that year (runs in a worker)."""
    kind, design, matrix, labels = job["kind"], job["design"], job["matrix"], job["labels"]
    dates, year, mcfg = job["dates"], job["year"], job["mcfg"]
    window = int(mcfg["window"])
    test_rows = np.flatnonzero((dates.year == year) & (np.arange(len(dates)) >= window - 1))
    if len(test_rows) == 0:
        return None
    rows = train_rows_before(int(test_rows[0]), window, labels.span)
    if len(rows) < 250:
        log.warning("year %d skipped: only %d training rows", year, len(rows))
        return None
    started = time.time()
    models, scaler, scaled = _fit_ensemble(kind, design, matrix, labels, rows, mcfg)
    reference = int(mcfg.get("reference_window", 250))
    frame = predict_with_reference(models, scaled, test_rows, window, reference, labels.horizons)
    frame.index = dates[test_rows]
    for j, h in enumerate(labels.horizons):
        frame[f"base_p_{h}"] = float(np.nanmean(labels.up[rows, j]))
        frame[f"base_mu_{h}"] = float(np.nanmean(np.clip(labels.z[rows, j], -5, 5)))
        frame[f"ret_{h}"] = labels.ret[test_rows, j]
        frame[f"up_{h}"] = labels.up[test_rows, j]
    # what the model would have said with each feature group neutralised (set to its training mean)
    for group, cols in job.get("group_columns", {}).items():
        neutral = scaled.copy()
        neutral[:, cols] = 0.0
        prob_without, _ = predict(models, make_windows(neutral, test_rows, window))
        for j, h in enumerate(labels.horizons):
            frame[f"p_{h}_wo_{group}"] = prob_without[:, j]
    frame["fold"] = year
    frame["train_rows"] = len(rows)
    frame.attrs["note"] = (f"{kind} {year}: trained on {len(rows)} rows (to {dates[rows[-1]].date()}), "
                           f"{len(test_rows)} test days, {time.time() - started:.0f}s")
    if job.get("save_path"):
        _save_ensemble(Path(job["save_path"]), kind, design, labels, rows, mcfg, models, scaler,
                       trained_through=str(dates[rows[-1]].date()), fold=year)
    return frame


def _save_ensemble(path: Path, kind: str, design: Design, labels: Labels, rows: np.ndarray, mcfg: dict,
                   models: list[nn.Module], scaler: Scaler, trained_through: str, fold: int) -> None:
    payload = {
        "kind": kind,
        "columns": design.columns,
        "branch_index": design.branch_index,
        "horizons": labels.horizons,
        "model_cfg": {"window": mcfg["window"], "hidden": mcfg["hidden"], "dropout": mcfg["dropout"],
                      "reference_window": int(mcfg.get("reference_window", 250))},
        "scaler": scaler.to_dict(),
        "state_dicts": [m.state_dict() for m in models],
        "base_p": [float(np.nanmean(labels.up[rows, j])) for j in range(len(labels.horizons))],
        "base_mu": [float(np.nanmean(np.clip(labels.z[rows, j], -5, 5))) for j in range(len(labels.horizons))],
        "trained_through": trained_through,
        "fold": fold,
        "parameters": count_parameters(models[0]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def walk_forward(bundle: PanelBundle, cfg: dict, *, kind: str = "cnn", groups: list[str] | None = None,
                 years: list[int] | None = None, workers: int | None = None,
                 save_last: Path | None = None) -> pd.DataFrame:
    """Out-of-sample predictions for every test year. One row per trading day.

    With save_last, the ensemble of the most recent fold is stored: it is the model that keeps
    predicting until the next yearly retrain, i.e. the one the paper account trades on.
    """
    mcfg = cfg["model"]
    horizons = [int(h) for h in mcfg["horizons"]]
    labels = make_labels(bundle.price, horizons, int(mcfg["execution_lag"]))
    design = Design.from_bundle(bundle, groups)
    matrix = bundle.panel[design.columns].to_numpy(dtype=float)
    dates = bundle.panel.index
    if years is None:
        years = list(range(int(cfg["walkforward"]["first_test_year"]), dates[-1].year + 1))
    present = sorted({bundle.groups[c] for c in design.columns})
    group_columns = {g: [i for i, c in enumerate(design.columns) if bundle.groups[c] == g] for g in present}
    if len(present) < 2:
        group_columns = {}
    jobs = [{"kind": kind, "design": design, "matrix": matrix, "labels": labels, "dates": dates,
             "year": year, "mcfg": mcfg, "group_columns": group_columns} for year in years]
    if save_last is not None and jobs:
        jobs[-1]["save_path"] = str(save_last)

    if workers is None:
        workers = min(len(jobs), os.cpu_count() or 1)
    if workers > 1 and kind != "linear":
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker) as pool:
            frames = []
            for frame in pool.map(_run_fold, jobs):
                if frame is not None:
                    log.info(frame.attrs.get("note", ""))
                    frames.append(frame)
    else:
        frames = []
        for job in jobs:
            frame = _run_fold(job)
            if frame is not None:
                log.info(frame.attrs.get("note", ""))
                frames.append(frame)
    if not frames:
        raise RuntimeError("no walk-forward folds could be built")
    return pd.concat(frames)


# ---------------------------------------------------------------------------------------------
# Production ensemble = the latest fold's models, used by the forward paper account
# ---------------------------------------------------------------------------------------------

def load_production(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    design = Design(columns=payload["columns"], branch_index=payload["branch_index"])
    models = []
    for state in payload["state_dicts"]:
        model = build_model(payload["kind"], design, len(payload["horizons"]), payload["model_cfg"])
        model.load_state_dict(state)
        models.append(model.eval())
    payload["models"] = models
    payload["scaler_obj"] = Scaler.from_dict(payload["scaler"])
    return payload


def predict_production(production: dict, bundle: PanelBundle, after=None) -> pd.DataFrame:
    """Predictions of the production ensemble for panel rows dated after `after` (default: all)."""
    window = int(production["model_cfg"]["window"])
    reference = int(production["model_cfg"].get("reference_window", 250))
    missing = [c for c in production["columns"] if c not in bundle.panel.columns]
    if missing:
        raise RuntimeError(f"panel is missing columns the model was trained on: {missing[:5]}")
    matrix = bundle.panel[production["columns"]].to_numpy(dtype=float)
    scaled = production["scaler_obj"].transform(matrix)
    rows = np.arange(window - 1, len(matrix))
    if after is not None:
        rows = rows[bundle.panel.index[rows] > pd.Timestamp(after)]
    if len(rows) == 0:
        return pd.DataFrame()
    frame = predict_with_reference(production["models"], scaled, rows, window, reference, production["horizons"])
    frame.index = bundle.panel.index[rows]
    for j, h in enumerate(production["horizons"]):
        frame[f"base_p_{h}"] = production["base_p"][j]
        frame[f"base_mu_{h}"] = production["base_mu"][j]
    frame["fold"] = production["fold"]
    return frame
