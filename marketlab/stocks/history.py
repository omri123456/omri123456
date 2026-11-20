"""Index members and their daily history, fetched with the day-trader skill's own Yahoo client.

Survivorship: the members are today's. Stocks that left the index (or failed) are missing from
the past, so past breadth is computed on survivors. Read every historical number with that in
mind; the bias flatters the past a little.
"""

from __future__ import annotations

import gzip
import importlib.util
import logging
import re
import time
from pathlib import Path

import pandas as pd

from ..config import Paths
from ..data import http

log = logging.getLogger(__name__)

MEMBERS_URL = "https://en.wikipedia.org/w/index.php?title=List_of_NASDAQ-100_companies&action=raw"
_ROW = re.compile(r"^\|\s*([A-Z][A-Z.\-]{0,6})\s*\|\|\s*\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", re.M)


def stocks_dir(paths: Paths) -> Path:
    return paths.raw / "stocks"


def parse_members(wikitext: str) -> pd.DataFrame:
    """Rows of the 'constituents' table: ticker, company, industry."""
    start = wikitext.find('id="constituents"')
    table = wikitext[start:wikitext.find("|}", start)] if start >= 0 else wikitext
    rows = []
    for line in table.split("|-"):
        cells = [c.strip() for c in line.strip().lstrip("|").split("||")]
        if len(cells) < 4 or not re.fullmatch(r"[A-Z][A-Z.\-]{0,6}", cells[0]):
            continue
        name = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", cells[1])
        rows.append({"symbol": cells[0], "company": re.sub(r"\s+", " ", name).strip(),
                     "industry": re.sub(r"<ref.*", "", cells[3]).strip()})
    return pd.DataFrame(rows).drop_duplicates("symbol").reset_index(drop=True)


def load_members(paths: Paths) -> pd.DataFrame | None:
    path = stocks_dir(paths) / "members.csv"
    return pd.read_csv(path) if path.exists() else None


def _scanner(paths: Paths):
    """The skill's scan.py as a module: its Yahoo client, parsers and rules."""
    path = paths.root / ".claude" / "skills" / "trader-scanner" / "scan.py"
    if not path.exists():
        raise RuntimeError("the day-trader skill pack is not installed in .claude/skills")
    spec = importlib.util.spec_from_file_location("trader_scan", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_frame(scan, data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """{symbol: spark series} -> (closes, volumes) indexed by New York trading date."""
    closes, volumes = {}, {}
    for sym, ser in data.items():
        rows = {}
        for ts, c, v in zip(ser.get("t", []), ser.get("c", []), ser.get("v", []) or [None] * len(ser.get("t", []))):
            if ts is None or c is None:
                continue
            rows[pd.Timestamp(scan.ts_to_eastern(ts).date())] = (float(c), float(v) if v is not None else float("nan"))
        if rows:
            frame = pd.DataFrame.from_dict(rows, orient="index", columns=["c", "v"]).sort_index()
            closes[sym], volumes[sym] = frame["c"], frame["v"]
    return pd.DataFrame(closes).sort_index(), pd.DataFrame(volumes).sort_index()


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(path, "wb", mtime=0) as fh:
        fh.write(frame.to_csv(float_format="%.6g").encode("utf-8"))


def load_history(paths: Paths) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    out = []
    for name in ("closes", "volumes"):
        path = stocks_dir(paths) / f"{name}.csv.gz"
        out.append(pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None)
    return out[0], out[1]


def fetch(paths: Paths, *, max_age_hours: float = 12.0, pause: float = 0.5) -> dict:
    """Members (weekly) and daily bars: 20 years the first time, the last 3 months after that."""
    folder = stocks_dir(paths)
    folder.mkdir(parents=True, exist_ok=True)
    members_path = folder / "members.csv"
    if not http.is_fresh(members_path, 7 * 24.0):
        members = parse_members(http.get(MEMBERS_URL).decode("utf-8", "replace"))
        if len(members) < 90:
            raise RuntimeError(f"members table looks wrong: {len(members)} rows")
        members.to_csv(members_path, index=False)
    members = pd.read_csv(members_path)
    closes_path = folder / "closes.csv.gz"
    if http.is_fresh(closes_path, max_age_hours):
        return {"members": len(members), "status": "cached"}

    scan = _scanner(paths)
    old_closes, old_volumes = load_history(paths)
    have = set(old_closes.columns) if old_closes is not None else set()
    symbols = list(members["symbol"])
    jobs = [("3mo", [s for s in symbols if s in have]), ("20y", [s for s in symbols if s not in have])]
    data: dict = {}
    failed: list[str] = []
    for rng, syms in jobs:
        for i in range(0, len(syms), scan.SPARK_MAX):
            batch = syms[i:i + scan.SPARK_MAX]
            try:
                data.update(scan.parse_spark(scan.yahoo_get(scan.spark_path(batch, rng, "1d")),
                                             {scan.to_yahoo(s): s for s in batch}))
            except Exception as exc:  # one bad batch must not lose the rest
                log.warning("stock history batch failed (%s): %s", rng, exc)
                failed.extend(batch)
            time.sleep(pause)
    closes, volumes = _to_frame(scan, data)
    if old_closes is not None:
        closes = closes.combine_first(old_closes) if len(closes) else old_closes
        volumes = volumes.combine_first(old_volumes) if len(volumes) else old_volumes
    keep = [s for s in symbols if s in closes.columns]
    closes, volumes = closes[keep].sort_index(), volumes.reindex(columns=keep).sort_index()
    _write(closes, closes_path)
    _write(volumes, folder / "volumes.csv.gz")
    return {"members": len(members), "with_history": len(keep), "failed": failed,
            "first": str(closes.index[0].date()), "last": str(closes.index[-1].date())}
