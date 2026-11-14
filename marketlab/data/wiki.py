"""Wikipedia "Portal:Current events": a dated, human-curated log of world events.

One rendered page per month is downloaded (about 280 requests for 23 years, paced one per
second) and parsed into one row per event: date, category (where the page has one), text.
"""

from __future__ import annotations

import calendar
import gzip
import logging
import os
import re
from datetime import date
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from . import http

log = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
_EMPTY_PARENS = re.compile(r"\(\s*[,;]?\s*\)")
_BRACKET_NOTE = re.compile(r"\[\s*(permanent dead link|dead link|citation needed|\d+)\s*\]", re.I)


def month_iter(start: str, end: date | None = None):
    """Yield (year, month) from 'YYYY-MM' up to and including the month of `end` (default today)."""
    end = end or date.today()
    year, month = (int(x) for x in start.split("-")[:2])
    while (year, month) <= (end.year, end.month):
        yield year, month
        month += 1
        if month == 13:
            year, month = year + 1, 1


def cache_path(wiki_dir: Path, year: int, month: int) -> Path:
    return wiki_dir / f"{year:04d}-{month:02d}.html.gz"


def fetch_month(url_template: str, wiki_dir: Path, year: int, month: int,
                today: date | None = None) -> bool:
    """Download one month page (gzip-cached). Old months are never re-downloaded."""
    today = today or date.today()
    path = cache_path(wiki_dir, year, month)
    months_old = (today.year - year) * 12 + (today.month - month)
    max_age = None if months_old >= 2 else 12.0  # recent months still receive edits
    if http.is_fresh(path, max_age):
        return False
    url = url_template.format(month=calendar.month_name[month], year=year)
    body = http.get(url, timeout=90.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.part")
    with gzip.open(tmp, "wb") as fh:
        fh.write(body)
    tmp.replace(path)
    return True


def fetch_all(url_template: str, wiki_dir: Path, start: str) -> dict[str, int]:
    counts = {"downloaded": 0, "cached": 0, "failed": 0}
    for year, month in month_iter(start):
        try:
            key = "downloaded" if fetch_month(url_template, wiki_dir, year, month) else "cached"
        except http.FetchError as exc:
            key = "failed"
            log.warning("wiki %04d-%02d failed: %s", year, month, exc)
        counts[key] += 1
    return counts


def _clean(text: str) -> str:
    text = _BRACKET_NOTE.sub(" ", text)
    text = _EMPTY_PARENS.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text.strip(" .;:,") if len(text) < 4 else text


def _own_text(li) -> str:
    """Text of an <li> without its nested lists (those are parsed as separate events)."""
    parts = []
    for child in li.children:
        name = getattr(child, "name", None)
        if name in ("ul", "ol", "dl"):
            continue
        parts.append(child.get_text(" ") if name else str(child))
    return _clean(" ".join(parts))


def _walk_list(ul, context: str, out: list[str]) -> None:
    for li in ul.find_all("li", recursive=False):
        own = _own_text(li)
        nested = li.find_all(["ul", "ol"], recursive=False)
        if nested:
            # A parent bullet is a running story ("Russo-Ukrainian War"); keep it as context.
            child_context = f"{context} {own}".strip() if own else context
            for sub in nested:
                _walk_list(sub, child_context[-200:], out)
        elif own:
            out.append(f"{context}: {own}" if context else own)


def parse_month(html: str) -> list[dict]:
    """Parse one rendered month page into event dicts: date, category, text."""
    soup = BeautifulSoup(html, "lxml")
    # Source citations "(Reuters)" are external links; drop them so they do not count as content.
    for link in soup.select("a.external"):
        link.decompose()
    for junk in soup.select("sup, style, script, .mw-editsection, .noprint"):
        junk.decompose()
    events: list[dict] = []
    for day in soup.select("div.current-events-main"):
        bday = day.select_one("span.bday")
        if bday is None:
            continue
        try:
            day_date = pd.Timestamp(bday.get_text(strip=True)).normalize()
        except ValueError:
            continue
        content = day.select_one("div.current-events-content") or day
        category = ""
        for child in content.find_all(recursive=False):
            if child.name in ("ul", "ol"):
                texts: list[str] = []
                _walk_list(child, "", texts)
                for text in texts:
                    if len(text) >= 15:
                        events.append({"date": day_date, "category": category, "text": text})
            else:
                heading = _clean(child.get_text(" "))
                if 0 < len(heading) <= 60:
                    category = heading
    return events


def parse_all(wiki_dir: Path) -> pd.DataFrame:
    """Parse every cached month into one DataFrame sorted by date."""
    rows: list[dict] = []
    for path in sorted(wiki_dir.glob("*.html.gz")):
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
                month_events = parse_month(fh.read())
        except Exception as exc:
            log.warning("could not parse %s: %s", path.name, exc)
            continue
        if not month_events:
            log.info("no dated events found in %s", path.name)
        rows.extend(month_events)
    df = pd.DataFrame(rows, columns=["date", "category", "text"])
    if not df.empty:
        df = df.drop_duplicates(subset=["date", "text"]).sort_values("date", kind="stable")
        df = df.reset_index(drop=True)
    return df
