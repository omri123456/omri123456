"""FRED time series via the public CSV download (no API key)."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd

from . import http

log = logging.getLogger(__name__)

CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"


def fetch_series(sid: str, dest_dir: Path, *, start: str = "1995-01-01",
                 max_age_hours: float | None = 12.0) -> bool:
    """Download one series to dest_dir/<sid>.csv. Returns True if a download happened."""
    return http.fetch_to_file(CSV_URL.format(sid=sid, start=start), dest_dir / f"{sid}.csv",
                              max_age_hours=max_age_hours)


def parse_series(raw: bytes | str, sid: str) -> pd.Series:
    """Parse a fredgraph CSV body into a float Series indexed by observation date."""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] < 2:
        raise ValueError(f"unexpected FRED payload for {sid}: columns {list(df.columns)}")
    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    values = pd.to_numeric(df.iloc[:, 1], errors="coerce")  # "." or blank marks a missing value
    series = pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(dates), name=sid)
    series = series[series.index.notna()].dropna().sort_index()
    return series[~series.index.duplicated(keep="last")]


def load_series(sid: str, src_dir: Path) -> pd.Series | None:
    path = src_dir / f"{sid}.csv"
    if not path.exists():
        return None
    try:
        return parse_series(path.read_bytes(), sid)
    except Exception as exc:  # a corrupt cache file should not kill the pipeline
        log.warning("could not parse %s: %s", path, exc)
        return None


def fetch_all(series_ids: list[str], dest_dir: Path, *, max_age_hours: float | None = 12.0
              ) -> dict[str, str]:
    """Fetch every series; returns {sid: 'downloaded' | 'cached' | 'failed: ...'}."""
    status: dict[str, str] = {}
    for sid in series_ids:
        try:
            status[sid] = "downloaded" if fetch_series(sid, dest_dir, max_age_hours=max_age_hours) else "cached"
        except http.FetchError as exc:
            status[sid] = f"failed: {exc}"
            log.warning("FRED %s failed: %s", sid, exc)
    return status
