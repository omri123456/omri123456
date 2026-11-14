"""Newspaper-derived daily indices that are not on FRED.

- SF Fed Daily News Sentiment Index (Shapiro, Sudhof & Wilson): tone of economic news, 1980-.
- Geopolitical Risk index (Caldara & Iacoviello): share of articles on geopolitical threats
  and acts, 1985-.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_news_sentiment(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_excel(path, sheet_name="Data", engine="openpyxl")
    df.columns = [str(c).strip().lower() for c in df.columns]
    value_col = next(c for c in df.columns if c != "date")
    series = pd.Series(pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float),
                       index=pd.DatetimeIndex(pd.to_datetime(df["date"], errors="coerce")),
                       name="news_sentiment")
    series = series[series.index.notna()].dropna().sort_index()
    return series[~series.index.duplicated(keep="last")]


def load_gpr(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_excel(path, engine="xlrd")
    df.columns = [str(c).strip().upper() for c in df.columns]
    dates = pd.to_datetime(df["DAY"].astype(str).str.slice(0, 8), format="%Y%m%d", errors="coerce")
    out = pd.DataFrame({
        "gpr": pd.to_numeric(df["GPRD"], errors="coerce").to_numpy(dtype=float),
        "gpr_act": pd.to_numeric(df["GPRD_ACT"], errors="coerce").to_numpy(dtype=float),
        "gpr_threat": pd.to_numeric(df["GPRD_THREAT"], errors="coerce").to_numpy(dtype=float),
    }, index=pd.DatetimeIndex(dates))
    out = out[out.index.notna()].dropna(how="all").sort_index()
    return out[~out.index.duplicated(keep="last")]
