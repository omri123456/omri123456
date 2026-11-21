"""Fetch every source into the local cache and load it back as one RawData bundle."""

from __future__ import annotations

import gzip
from datetime import date
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
    stock_closes: pd.DataFrame | None = None  # NASDAQ-100 members, daily closes (dates x symbols)
    provisional: dict[str, list[str]] = field(default_factory=dict)  # series -> dates filled from Yahoo

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
            events=events, stock_closes=cut(self.stock_closes),
        )


def events_dir(paths: Paths) -> Path:
    """One gzip CSV of parsed events per year. Kept in git: past years never change, so a fresh
    container only has to download the current months instead of 280 Wikipedia pages."""
    return paths.root / "data" / "events"


def _year_file(paths: Paths, year: int) -> Path:
    return events_dir(paths) / f"wiki_{year}.csv.gz"


def _read_year(paths: Paths, year: int) -> pd.DataFrame:
    path = _year_file(paths, year)
    if not path.exists():
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "category": pd.Series(dtype=object),
                             "text": pd.Series(dtype=object)})
    return pd.read_csv(path, parse_dates=["date"], keep_default_na=False)


def _months_with_events(paths: Paths) -> set[tuple[int, int]]:
    months: set[tuple[int, int]] = set()
    for path in events_dir(paths).glob("wiki_*.csv.gz"):
        dates = pd.read_csv(path, usecols=["date"], parse_dates=["date"])["date"]
        months.update(zip(dates.dt.year, dates.dt.month))
    return months


def _update_events(paths: Paths, changed: list[tuple[int, int]]) -> int:
    """Re-parse the months that changed and rewrite the year files they belong to."""
    by_year: dict[int, list[int]] = {}
    for year, month in changed:
        by_year.setdefault(year, []).append(month)
    for year, months in sorted(by_year.items()):
        df = _read_year(paths, year)
        df = df[~(df["date"].dt.year.eq(year) & df["date"].dt.month.isin(months))]
        fresh = []
        for month in months:
            with gzip.open(wiki.cache_path(paths.wiki, year, month), "rt", encoding="utf-8", errors="replace") as fh:
                fresh.append(pd.DataFrame(wiki.parse_month(fh.read()), columns=["date", "category", "text"]))
        df = pd.concat([df, *fresh], ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"].dt.year == year]  # a page can list a day of a neighbouring month
        df = df.drop_duplicates(subset=["date", "text"]).sort_values("date", kind="stable")
        events_dir(paths).mkdir(parents=True, exist_ok=True)
        # mtime=0 keeps the gzip bytes identical when the content is, so git sees no change
        with gzip.GzipFile(_year_file(paths, year), "wb", mtime=0) as fh:
            fh.write(df.to_csv(index=False).encode("utf-8"))
    return sum(len(_read_year(paths, y)) for y in by_year)


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
    known = _months_with_events(paths)
    today = date.today()
    for year, month in wiki.month_iter(cfg["project"]["wiki_start"]):
        settled = (today.year - year) * 12 + (today.month - month) >= 2
        if settled and (year, month) in known and not wiki.cache_path(paths.wiki, year, month).exists():
            wiki_counts["cached"] += 1  # parsed events are already stored for this finished month
            continue
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
    if changed:
        summary["wiki_events_updated"] = _update_events(paths, changed)
    return summary


def provisional_path(paths: Paths) -> Path:
    return paths.fred / "provisional_closes.csv"


def fetch_provisional(cfg: dict, paths: Paths) -> dict:
    """Closes of the last few sessions from Yahoo for the series FRED may post late."""
    wanted = cfg["project"].get("provisional_closes") or {}
    if not wanted:
        return {}
    from ..stocks.history import _scanner  # the day-trader skill's Yahoo client
    scan = _scanner(paths)
    data = scan.parse_spark(scan.yahoo_get(scan.spark_path(list(wanted.values()), "5d", "1d")))
    rows = []
    for sid, symbol in wanted.items():
        series = data.get(symbol) or data.get(symbol.lstrip("^")) or {}
        closes = {}
        for ts, close in zip(series.get("t", []), series.get("c", [])):
            if ts is not None and close is not None:
                closes[str(scan.ts_to_eastern(ts).date())] = float(close)  # one bar a day; a late tick wins
        # only finished sessions: the bar of a day that is still trading is not a close
        now = scan.to_eastern(pd.Timestamp.now(tz="UTC").to_pydatetime())
        session = scan.session(now.date())
        for day, close in closes.items():
            if day == str(now.date()) and session is not None and now.time() < session[1]:
                continue
            rows.append({"series": sid, "date": day, "close": close})
    frame = pd.DataFrame(rows, columns=["series", "date", "close"])
    frame.to_csv(provisional_path(paths), index=False)
    return {sid: frame.loc[frame["series"] == sid, "date"].max() for sid in wanted}


def load_raw(cfg: dict, paths: Paths) -> RawData:
    raw = RawData()
    stock_file = paths.raw / "stocks" / "closes.csv.gz"
    if stock_file.exists():
        raw.stock_closes = pd.read_csv(stock_file, index_col=0, parse_dates=True)
    for sid in cfg["fred"]:
        series = fred.load_series(sid, paths.fred)
        if series is not None and len(series):
            raw.fred[sid] = series
        else:
            log.warning("FRED series %s is not in the cache; features that need it are skipped", sid)
    if provisional_path(paths).exists():
        extra = pd.read_csv(provisional_path(paths), parse_dates=["date"])
        for sid, part in extra.groupby("series"):
            if sid not in raw.fred:
                continue
            newer = part[part["date"] > raw.fred[sid].index.max()]
            if len(newer):  # FRED has not posted these days yet
                add = pd.Series(newer["close"].to_numpy(float), index=pd.DatetimeIndex(newer["date"]), name=sid)
                raw.fred[sid] = pd.concat([raw.fred[sid], add]).sort_index()
                raw.provisional[sid] = [str(d.date()) for d in add.index]
    raw.sentiment = _safe(indices.load_news_sentiment, paths.misc / "news_sentiment.xlsx")
    raw.gpr = _safe(indices.load_gpr, paths.misc / "gpr_daily.xls")
    raw.spf = _safe(spf.load_spf, paths.misc / "spf_meanlevel.xlsx")
    raw.fomc = _safe(fomc.load_fomc, paths.fomc)
    files = sorted(events_dir(paths).glob("wiki_*.csv.gz"))
    if files:
        raw.events = pd.concat([pd.read_csv(f, parse_dates=["date"], keep_default_na=False) for f in files],
                               ignore_index=True).sort_values("date", kind="stable").reset_index(drop=True)
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
