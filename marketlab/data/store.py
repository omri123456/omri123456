"""Fetch every source into the local cache and load it back as one RawData bundle."""

from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import Paths
from . import fomc, fred, http, indices, spf, wiki

log = logging.getLogger(__name__)


@dataclass
class RawData:
    """Everything the feature builder needs, exactly as observed (no alignment, no lags)."""

    fred: dict[str, pd.Series] = field(default_factory=dict)
    sentiment: pd.Series | None = None
    gpr: pd.DataFrame | None = None
    spf: pd.DataFrame | None = None
    fomc: pd.DataFrame | None = None
    events: pd.DataFrame | None = None  # columns: date, category, text

    def truncate(self, last_date) -> "RawData":
        """Copy of the bundle without any observation dated after last_date (for leak tests)."""
        last = pd.Timestamp(last_date)

        def cut(obj):
            if obj is None:
                return None
            return obj[obj.index <= last]

        events = None
        if self.events is not None:
            events = self.events[self.events["date"] <= last].reset_index(drop=True)
        return RawData(
            fred={k: cut(v) for k, v in self.fred.items()},
            sentiment=cut(self.sentiment), gpr=cut(self.gpr), spf=cut(self.spf),
            fomc=self.fomc,  # the meeting schedule is published ahead of time
            events=events,
        )


def events_path(paths: Paths) -> Path:
    return paths.processed / "wiki_events.csv.gz"


def _update_events(paths: Paths, changed: list[tuple[int, int]]) -> int:
    """Re-parse only the months that changed and merge them into the events file."""
    target = events_path(paths)
    if not target.exists():
        df = wiki.parse_all(paths.wiki)
    else:
        df = pd.read_csv(target, parse_dates=["date"], keep_default_na=False)
        for year, month in changed:
            in_month = (df["date"].dt.year == year) & (df["date"].dt.month == month)
            df = df[~in_month]
            path = wiki.cache_path(paths.wiki, year, month)
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
                fresh = pd.DataFrame(wiki.parse_month(fh.read()), columns=["date", "category", "text"])
            df = pd.concat([df, fresh], ignore_index=True)
        df = df.drop_duplicates(subset=["date", "text"]).sort_values("date", kind="stable")
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False, compression="gzip")
    return len(df)


def fetch_everything(cfg: dict, paths: Paths, *, max_age_hours: float | None = 12.0) -> dict:
    """Download/refresh all sources. Failures are reported, not fatal: the pipeline uses what it has."""
    paths.ensure()
    summary: dict = {}

    fred_status = fred.fetch_all(list(cfg["fred"].keys()), paths.fred, max_age_hours=max_age_hours)
    summary["fred_failed"] = sorted(k for k, v in fred_status.items() if v.startswith("failed"))
    summary["fred_ok"] = len(fred_status) - len(summary["fred_failed"])

    src = cfg["sources"]
    for name, filename in (("news_sentiment", "news_sentiment.xlsx"), ("gpr", "gpr_daily.xls"),
                           ("spf", "spf_meanlevel.xlsx")):
        try:
            http.fetch_to_file(src[name]["url"], paths.misc / filename, max_age_hours=max_age_hours)
            summary[name] = "ok"
        except http.FetchError as exc:
            summary[name] = f"failed: {exc}"
            log.warning("%s failed: %s", name, exc)

    summary["fomc"] = fomc.fetch_all(src["fomc"], paths.fomc)

    changed: list[tuple[int, int]] = []
    wiki_counts = {"downloaded": 0, "cached": 0, "failed": 0}
    for year, month in wiki.month_iter(cfg["project"]["wiki_start"]):
        try:
            if wiki.fetch_month(src["wiki"]["url"], paths.wiki, year, month):
                changed.append((year, month))
                wiki_counts["downloaded"] += 1
            else:
                wiki_counts["cached"] += 1
        except http.FetchError as exc:
            wiki_counts["failed"] += 1
            log.warning("wiki %04d-%02d failed: %s", year, month, exc)
    summary["wiki"] = wiki_counts
    if changed or not events_path(paths).exists():
        summary["wiki_events"] = _update_events(paths, changed)
    return summary


def load_raw(cfg: dict, paths: Paths) -> RawData:
    raw = RawData()
    for sid in cfg["fred"]:
        series = fred.load_series(sid, paths.fred)
        if series is not None and len(series):
            raw.fred[sid] = series
        else:
            log.warning("FRED series %s is not in the cache; features that need it are skipped", sid)
    raw.sentiment = _safe(indices.load_news_sentiment, paths.misc / "news_sentiment.xlsx")
    raw.gpr = _safe(indices.load_gpr, paths.misc / "gpr_daily.xls")
    raw.spf = _safe(spf.load_spf, paths.misc / "spf_meanlevel.xlsx")
    raw.fomc = _safe(fomc.load_fomc, paths.fomc)
    if events_path(paths).exists():
        raw.events = pd.read_csv(events_path(paths), parse_dates=["date"], keep_default_na=False)
    return raw


def _safe(loader, path: Path):
    try:
        return loader(path)
    except Exception as exc:
        log.warning("could not load %s: %s", path, exc)
        return None


def freshness(raw: RawData) -> pd.DataFrame:
    """Last observation date of every source: used to sanity-check the configured lags."""
    rows = [{"source": f"fred:{k}", "last_obs": v.index.max(), "n": len(v)} for k, v in raw.fred.items()]
    for name, obj in (("news_sentiment", raw.sentiment), ("gpr", raw.gpr), ("spf", raw.spf)):
        if obj is not None and len(obj):
            rows.append({"source": name, "last_obs": obj.index.max(), "n": len(obj)})
    if raw.events is not None and len(raw.events):
        rows.append({"source": "wiki_events", "last_obs": raw.events["date"].max(), "n": len(raw.events)})
    if raw.fomc is not None and len(raw.fomc):
        rows.append({"source": "fomc", "last_obs": raw.fomc.index.max(), "n": len(raw.fomc)})
    return pd.DataFrame(rows)
