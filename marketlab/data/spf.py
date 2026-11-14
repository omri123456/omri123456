"""Survey of Professional Forecasters (Philadelphia Fed): quarterly consensus forecasts.

Column convention in the workbook: X1 = last quarter (already known), X2 = the survey quarter,
X3..X6 = one to four quarters ahead. We turn the levels into the forward-looking quantities a
market participant cares about and into the revision against the previous survey.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _sheet(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=name, engine="openpyxl")
    df.columns = [str(c).strip().upper() for c in df.columns]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["YEAR", "QUARTER"])
    df["YEAR"] = df["YEAR"].astype(int)
    df["QUARTER"] = df["QUARTER"].astype(int)
    return df.set_index(["YEAR", "QUARTER"]).sort_index()


def survey_available_date(year: int, quarter: int) -> pd.Timestamp:
    """Conservative availability: last day of the survey quarter's middle month.

    Results are normally published around the middle of that month; month-end leaves room for
    the occasional late release.
    """
    middle_month = 3 * (quarter - 1) + 2
    return pd.Timestamp(year=year, month=middle_month, day=1) + pd.offsets.MonthEnd(0)


def load_spf(path: Path) -> pd.DataFrame | None:
    """Return forecast features indexed by the date each survey became available."""
    if not path.exists():
        return None
    rgdp, unemp, cpi = _sheet(path, "RGDP"), _sheet(path, "UNEMP"), _sheet(path, "CPI")
    tbill, tbond, cprof = _sheet(path, "TBILL"), _sheet(path, "TBOND"), _sheet(path, "CPROF")

    out = pd.DataFrame(index=rgdp.index)
    # expected real GDP growth over the next four quarters, in percent
    out["spf_gdp_growth"] = (rgdp["RGDP6"] / rgdp["RGDP2"] - 1.0) * 100.0
    # expected change in the unemployment rate over the next four quarters
    out["spf_unemp_chg"] = unemp["UNEMP6"] - unemp["UNEMP2"]
    # expected CPI inflation, average of the next four quarters (annualised quarterly rates)
    out["spf_cpi"] = cpi[["CPI3", "CPI4", "CPI5", "CPI6"]].mean(axis=1)
    # expected path of short and long rates
    out["spf_tbill_chg"] = tbill["TBILL6"] - tbill["TBILL2"]
    out["spf_tbond_chg"] = tbond["TBOND6"] - tbond["TBOND2"]
    # expected growth of corporate profits over the next four quarters, in percent
    out["spf_profit_growth"] = (cprof["CPROF6"] / cprof["CPROF2"] - 1.0) * 100.0
    out = out.replace([np.inf, -np.inf], np.nan)

    # revisions: how the same quantity changed since the previous survey
    for col in ["spf_gdp_growth", "spf_unemp_chg", "spf_cpi", "spf_profit_growth"]:
        out[col + "_rev"] = out[col].diff()

    out.index = pd.DatetimeIndex([survey_available_date(y, q) for y, q in out.index], name="available")
    return out.dropna(how="all").sort_index()
