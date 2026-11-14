"""FOMC meeting dates from federalreserve.gov.

Scheduled meetings are published about a year ahead, so "days until the next scheduled
decision" is information one really has in advance. Unscheduled meetings and conference calls
are only known once they happen and are flagged separately.
"""

from __future__ import annotations

import calendar
import logging
import re
from pathlib import Path

import pandas as pd

from . import http

log = logging.getLogger(__name__)

_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _month_number(token: str) -> int | None:
    return _MONTHS.get(token.strip().lower().rstrip("."))


def _decision_date(year: int, month_text: str, day_text: str) -> pd.Timestamp | None:
    """Last day of a meeting given e.g. ('January', '27-28') or ('Jan/Feb', '31-1')."""
    months = [m for m in (_month_number(t) for t in re.split(r"[/\-]", month_text)) if m]
    days = [int(d) for d in re.findall(r"\d{1,2}", day_text)]
    if not months or not days:
        return None
    month, day = months[-1], days[-1]
    if len(months) == 1 and len(days) > 1 and days[-1] < days[0]:
        month = month % 12 + 1  # e.g. "April 30-1": the meeting ends in the next month
        if month == 1:
            year += 1
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return None


def parse_calendar(html: str) -> list[dict]:
    """Parse the current multi-year calendar page (2021 onward at the time of writing)."""
    rows: list[dict] = []
    for block in re.split(r'<div class="panel panel-default">', html)[1:]:
        year_match = re.search(r"(\d{4})\s+FOMC Meetings", block)
        if not year_match:
            continue
        year = int(year_match.group(1))
        pattern = (r'fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>\s*</div>\s*'
                   r'<div class="fomc-meeting__date[^>]*>([^<]+)</div>')
        for month_text, day_text in re.findall(pattern, block):
            decision = _decision_date(year, month_text, day_text)
            if decision is None:
                continue
            note = day_text.lower()
            if "cancel" in note:
                continue
            scheduled = not any(k in note for k in ("unscheduled", "notation", "conference"))
            rows.append({"date": decision, "scheduled": scheduled})
    return rows


def parse_historical(html: str, year: int) -> list[dict]:
    """Parse a fomchistoricalYYYY page: headings like 'January 29-30 Meeting - 2008'."""
    rows: list[dict] = []
    for raw in re.findall(r"<h5[^>]*>(.*?)</h5>", html, flags=re.S):
        heading = _WS.sub(" ", _TAG.sub("", raw)).strip()
        match = re.match(r"([A-Za-z]+(?:[/\-][A-Za-z]+)?)\.?\s+(\d{1,2}(?:\s*[-–]\s*\d{1,2})?)\s*(.*)", heading)
        if not match:
            continue
        month_text, day_text, rest = match.groups()
        decision = _decision_date(year, month_text, day_text.replace("–", "-"))
        if decision is None:
            continue
        rest_lower = rest.lower()
        scheduled = ("meeting" in rest_lower and "unscheduled" not in rest_lower
                     and "conference call" not in rest_lower)
        rows.append({"date": decision, "scheduled": scheduled})
    return rows


def fetch_all(cfg: dict, fomc_dir: Path) -> dict[str, int]:
    counts = {"downloaded": 0, "cached": 0, "failed": 0}
    jobs = [(cfg["calendar_url"], fomc_dir / "calendar.html", 24.0)]
    first, last = cfg["historical_years"]
    for year in range(first, last + 1):
        jobs.append((cfg["historical_url"].format(year=year), fomc_dir / f"{year}.html", None))
    for url, path, max_age in jobs:
        try:
            key = "downloaded" if http.fetch_to_file(url, path, max_age_hours=max_age) else "cached"
        except http.FetchError as exc:
            key = "failed"
            log.warning("FOMC page failed: %s", exc)
        counts[key] += 1
    return counts


def load_fomc(fomc_dir: Path) -> pd.DataFrame | None:
    """All known meetings as a DataFrame indexed by decision date with a 'scheduled' flag."""
    rows: list[dict] = []
    for path in sorted(fomc_dir.glob("*.html")):
        html = path.read_text(encoding="utf-8", errors="replace")
        if path.stem == "calendar":
            rows.extend(parse_calendar(html))
        elif path.stem.isdigit():
            rows.extend(parse_historical(html, int(path.stem)))
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates(subset=["date"]).set_index("date").sort_index()
    return df
