"""Configuration loading and project paths."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("MARKETLAB_HOME", Path(__file__).resolve().parents[1]))
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


class Paths:
    """All on-disk locations, derived from one root so tests can relocate them."""

    def __init__(self, root: Path | str = ROOT):
        self.root = Path(root)
        self.raw = self.root / "data" / "raw"
        self.fred = self.raw / "fred"
        self.wiki = self.raw / "wiki"
        self.misc = self.raw / "misc"
        self.fomc = self.raw / "fomc"
        self.processed = self.root / "data" / "processed"
        self.runs = self.root / "runs"
        self.models = self.root / "models"
        self.state = self.root / "state"
        self.reports = self.root / "reports"

    def ensure(self) -> "Paths":
        for p in (self.fred, self.wiki, self.misc, self.fomc, self.processed,
                  self.runs, self.models, self.state, self.reports):
            p.mkdir(parents=True, exist_ok=True)
        return self


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Load the default config, optionally layering another YAML file and a dict on top."""
    with open(DEFAULT_CONFIG, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with open(path, encoding="utf-8") as fh:
            _deep_update(cfg, yaml.safe_load(fh) or {})
    if overrides:
        _deep_update(cfg, copy.deepcopy(overrides))
    return cfg
