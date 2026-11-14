"""Event study: what did the index do around each kind of event?

Events are located in event time (the day the news happened, no publication delay), because
the question here is descriptive: "how does the market behave around X", not "can I trade it".
For each event day we measure the cumulative log return from 5 days before to 20 days after,
minus the average drift of the index, and average across events of the same kind.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.store import RawData
from ..features import text
from ..features.align import bucket_to_calendar, surprise
from ..features.panel import PanelBundle, wiki_buckets

PRE, POST = 5, 20


def _decluster(flags: pd.Series, strength: pd.Series, gap: int = 5) -> list[int]:
    """Keep the strongest flagged day of every cluster (events at least `gap` rows apart)."""
    candidates = np.flatnonzero(flags.to_numpy())
    order = sorted(candidates, key=lambda i: -strength.iloc[i])
    chosen: list[int] = []
    for i in order:
        if all(abs(i - j) >= gap for j in chosen):
            chosen.append(int(i))
    return sorted(chosen)


def find_events(raw: RawData, bundle: PanelBundle, calendar: pd.DatetimeIndex,
                threshold: float = 3.0) -> dict[str, list[int]]:
    """Row positions (in `calendar`) of every event type."""
    events: dict[str, list[int]] = {}

    buckets = wiki_buckets(raw, calendar)
    if buckets is not None:
        n = buckets["n"].clip(lower=1.0)
        for topic in text.TOPIC_PATTERNS:
            score = surprise(buckets[topic] / n, floor=0.03)
            # require at least two matching events so one stray headline is not a "spike"
            flags = (score >= threshold) & (buckets[topic] >= 2)
            events[f"news: {topic}"] = _decluster(flags.fillna(False), score.fillna(0.0))
        deaths = surprise(np.log1p(buckets["deaths"]), floor=0.5)
        events["news: mass casualties"] = _decluster((deaths >= threshold).fillna(False), deaths.fillna(0.0))

    def spikes(series: pd.Series | None, name: str, sign: float = 1.0, smooth: int = 1) -> None:
        if series is None or series.empty:
            return
        daily = series.rolling(smooth, min_periods=1).mean() if smooth > 1 else series
        on_calendar = bucket_to_calendar(daily.to_frame("x"), calendar, "mean")["x"]
        score = sign * surprise(on_calendar, floor=0.05)
        events[name] = _decluster((score >= threshold).fillna(False), score.fillna(0.0))

    if raw.gpr is not None:
        spikes(np.log(raw.gpr["gpr"].clip(lower=1.0)), "index: geopolitical risk spike")
    spikes(_log(raw.fred.get("USEPUINDXD")), "index: policy uncertainty spike", smooth=3)
    spikes(_log(raw.fred.get("WLEMUINDXD")), "index: equity uncertainty spike", smooth=3)
    if raw.sentiment is not None:
        spikes(raw.sentiment.diff(7), "index: news sentiment drop", sign=-1.0)

    if raw.fomc is not None:
        for flag, name in ((True, "fed: scheduled FOMC decision"), (False, "fed: unscheduled FOMC action")):
            dates = raw.fomc.index[raw.fomc["scheduled"] == flag]
            pos = calendar.searchsorted(dates, side="left")
            pos = pos[(pos < len(calendar)) & (dates >= calendar[0]) & (dates <= calendar[-1])]
            events[name] = sorted(set(int(p) for p in pos))

    ret = np.log(bundle.price.reindex(calendar)).diff()
    events["market: one-day drop over 3%"] = _decluster((ret <= -0.03).fillna(False), -ret.fillna(0.0))
    vix = raw.fred.get("VIXCLS")
    if vix is not None:
        jump = np.log(vix.reindex(calendar).ffill()).diff()
        events["market: VIX jump over 20%"] = _decluster((jump >= np.log(1.2)).fillna(False), jump.fillna(0.0))
    return events


def _log(series: pd.Series | None) -> pd.Series | None:
    return None if series is None else np.log(series.clip(lower=1.0))


def study(raw: RawData, bundle: PanelBundle, *, start=None, threshold: float = 3.0
          ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (summary per event type, average abnormal path per event type)."""
    calendar = bundle.price.index
    if start is not None:
        calendar = calendar[calendar >= pd.Timestamp(start)]
    logp = np.log(bundle.price.reindex(calendar)).to_numpy()
    daily = np.diff(logp, prepend=np.nan)
    drift = np.nanmean(daily)
    abs_ret = np.abs(daily)
    events = find_events(raw, bundle, calendar, threshold)

    offsets = np.arange(-PRE, POST + 1)
    rows, paths = [], {}
    for name, positions in events.items():
        usable = [i for i in positions if i - PRE - 20 >= 1 and i + POST < len(logp)]
        if len(usable) < 8:
            continue
        idx = np.array(usable)
        # cumulative abnormal log return relative to the close of the event day
        path = np.stack([logp[idx + k] - logp[idx] - drift * k for k in offsets], axis=1)
        raw_after = {k: logp[idx + k] - logp[idx] for k in (1, 5, 20)}
        car = {k: path[:, PRE + k] for k in (1, 5, 20)}
        half = len(idx) // 2
        vol_after = np.array([abs_ret[i + 1:i + 6].mean() for i in idx])
        vol_before = np.array([abs_ret[i - 20:i].mean() for i in idx])
        rows.append({
            "event": name, "n": len(idx),
            "first": str(calendar[idx[0]].date()), "last": str(calendar[idx[-1]].date()),
            "day0": float(np.mean(logp[idx] - logp[idx - 1] - drift)),
            "pre5": float(np.mean(-path[:, 0])),
            "car1": float(car[1].mean()), "car5": float(car[5].mean()), "car20": float(car[20].mean()),
            "t5": float(car[5].mean() / (car[5].std(ddof=1) / np.sqrt(len(idx)))),
            "t20": float(car[20].mean() / (car[20].std(ddof=1) / np.sqrt(len(idx)))),
            "hit5": float((raw_after[5] > 0).mean()), "hit20": float((raw_after[20] > 0).mean()),
            "car5_first_half": float(car[5][:half].mean()), "car5_second_half": float(car[5][half:].mean()),
            "vol_ratio": float(np.mean(vol_after) / np.mean(vol_before)),
        })
        paths[name] = path.mean(axis=0)
    summary = pd.DataFrame(rows).set_index("event") if rows else pd.DataFrame()
    path_table = pd.DataFrame(paths, index=offsets)
    path_table.index.name = "day"
    return summary, path_table


def baseline_hit_rates(bundle: PanelBundle, start=None) -> dict:
    """Unconditional share of positive 5- and 20-day returns: the yardstick for the hit rates."""
    price = bundle.price if start is None else bundle.price[bundle.price.index >= pd.Timestamp(start)]
    logp = np.log(price)
    return {f"hit{k}": float((logp.shift(-k) - logp > 0)[:-k].mean()) for k in (5, 20)}
