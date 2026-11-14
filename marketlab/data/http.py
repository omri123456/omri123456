"""Small polite HTTP client: honest client name, per-host pacing, retries, file cache."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

# Honest client identification only: no browser impersonation, no personal contact details.
# By default requests announces itself as "python-requests/x.y" (FRED only answers client names
# it recognises). Wikimedia asks scripts for a descriptive name instead.
HOST_USER_AGENT = {
    "en.wikipedia.org": "marketlab/0.1 (research script; python-requests)",
}

# Minimum seconds between two requests to the same host.
HOST_INTERVAL = {
    "en.wikipedia.org": 1.0,
    "fred.stlouisfed.org": 0.4,
}
DEFAULT_INTERVAL = 0.5

_last_request: dict[str, float] = {}
_session: requests.Session | None = None


class FetchError(RuntimeError):
    pass


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def _pace(host: str) -> None:
    interval = HOST_INTERVAL.get(host, DEFAULT_INTERVAL)
    wait = _last_request.get(host, 0.0) + interval - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request[host] = time.monotonic()


def get(url: str, *, timeout: float = 45.0, retries: int = 2) -> bytes:
    """GET a URL and return the body. Retries transient failures; never retries a refusal."""
    host = urlparse(url).netloc
    headers = {"User-Agent": HOST_USER_AGENT[host]} if host in HOST_USER_AGENT else None
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        _pace(host)
        try:
            resp = _get_session().get(url, timeout=timeout, headers=headers)
        except requests.RequestException as exc:  # connection reset, timeout, proxy error
            last_error = exc
            if "403" in str(exc) or "407" in str(exc):
                raise FetchError(f"{host}: blocked by network policy ({exc})") from exc
        else:
            if resp.status_code == 200:
                return resp.content
            last_error = FetchError(f"{host}: HTTP {resp.status_code} for {url}")
            if resp.status_code in (400, 401, 403, 404, 407):
                raise last_error
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After", "")
                delay = float(retry_after) if retry_after.isdigit() else 20.0 * (attempt + 1)
                log.warning("%s asked us to slow down; waiting %.0fs", host, delay)
                time.sleep(min(delay, 120.0))
                continue
        time.sleep(2.0 * (attempt + 1))
    raise FetchError(f"giving up on {url}: {last_error}")


def is_fresh(path: Path, max_age_hours: float | None) -> bool:
    """True when the cached file exists and is younger than max_age_hours (None = forever)."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    if max_age_hours is None:
        return True
    return (time.time() - path.stat().st_mtime) < max_age_hours * 3600.0


def fetch_to_file(url: str, path: Path, *, max_age_hours: float | None = 12.0,
                  timeout: float = 60.0) -> bool:
    """Download url into path unless the cached copy is fresh. Returns True if downloaded."""
    if is_fresh(path, max_age_hours):
        return False
    body = get(url, timeout=timeout)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.part")
    tmp.write_bytes(body)
    tmp.replace(path)
    return True
