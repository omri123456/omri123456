"""Synthetic raw data so the tests need neither the network nor the local cache."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketlab.config import load_config
from marketlab.data.store import RawData

TOPIC_SENTENCES = [
    "An earthquake kills 120 people in the region.",
    "The Federal Reserve cuts interest rates by 50 basis points.",
    "Stock markets fall sharply on Wall Street amid recession fears.",
    "Voters go to the polls in a presidential election.",
    "Airstrikes hit the capital as troops advance; at least 30 dead.",
    "The company files for bankruptcy after a failed bailout.",
    "Protesters clash with police in the capital.",
    "Scientists announce a breakthrough and a new treaty is signed.",
    "The United States Senate approves a trade deal lifting tariffs.",
    "A new coronavirus outbreak prompts a quarantine.",
]


def _walk(rng, n, start=100.0, scale=0.01):
    return start * np.exp(np.cumsum(rng.normal(0.0002, scale, n)))


def make_raw(cfg: dict, start="2001-01-01", end="2011-12-30", seed=0) -> RawData:
    rng = np.random.default_rng(seed)
    business = pd.bdate_range(start, end)
    calendar_days = pd.date_range(start, end, freq="D")
    weekly = pd.date_range(start, end, freq="W-FRI")
    monthly = pd.date_range(start, end, freq="MS")
    raw = RawData()
    for sid, spec in cfg["fred"].items():
        index = {"d": business, "w": weekly, "m": monthly, "c": calendar_days}[spec["freq"]]
        if sid.startswith(("DGS", "DFF", "DFII", "T10", "T5", "BAA", "AAA")):
            values = 3.0 + np.cumsum(rng.normal(0, 0.03, len(index)))
        elif sid in ("NFCI", "STLFSI4"):
            values = rng.normal(0, 0.5, len(index))
        else:
            values = _walk(rng, len(index), scale=0.012 if spec["freq"] in ("d", "c") else 0.02)
        raw.fred[sid] = pd.Series(values, index=index, name=sid)
    raw.sentiment = pd.Series(np.cumsum(rng.normal(0, 0.01, len(calendar_days))), index=calendar_days)
    raw.gpr = pd.DataFrame({c: _walk(rng, len(calendar_days), 100.0, 0.05)
                            for c in ("gpr", "gpr_act", "gpr_threat")}, index=calendar_days)
    quarters = pd.date_range(start, end, freq="QS")
    available = pd.DatetimeIndex([q + pd.offsets.MonthBegin(1) + pd.offsets.MonthEnd(0) for q in quarters])
    columns = ["spf_gdp_growth", "spf_unemp_chg", "spf_cpi", "spf_tbill_chg", "spf_tbond_chg",
               "spf_profit_growth", "spf_gdp_growth_rev", "spf_unemp_chg_rev", "spf_cpi_rev",
               "spf_profit_growth_rev"]
    raw.spf = pd.DataFrame(rng.normal(0, 1, (len(available), len(columns))), index=available, columns=columns)
    fomc_dates, flags = [], []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 2):
        for month in (1, 3, 5, 6, 8, 9, 11, 12):
            fomc_dates.append(pd.Timestamp(year=year, month=month, day=15) + pd.offsets.BDay(0))
            flags.append(True)
        fomc_dates.append(pd.Timestamp(year=year, month=10, day=8) + pd.offsets.BDay(0))
        flags.append(False)
    raw.fomc = pd.DataFrame({"scheduled": flags}, index=pd.DatetimeIndex(fomc_dates)).sort_index()
    rows = []
    for day in calendar_days:
        for _ in range(int(rng.integers(2, 7))):
            rows.append({"date": day, "category": "", "text": TOPIC_SENTENCES[int(rng.integers(len(TOPIC_SENTENCES)))]})
    raw.events = pd.DataFrame(rows)
    return raw


@pytest.fixture(scope="session")
def cfg():
    return load_config(overrides={
        "project": {"start": "2003-01-01"},
        "model": {"window": 20, "epochs": 1, "seeds": [1], "hidden": 4, "horizons": [1, 5], "trade_horizon": 5,
                  "reference_window": 60},
    })


@pytest.fixture(scope="session")
def raw(cfg):
    return make_raw(cfg)


@pytest.fixture(scope="session")
def bundle(raw, cfg):
    from marketlab.features.panel import build_panel
    return build_panel(raw, cfg)
