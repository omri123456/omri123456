"""Assemble the daily point-in-time feature panel.

Row t = everything that can be known in time for an order filled at the close of trading day
t+1. Columns are grouped (market, rates, cross, liquidity, macro, forecasts, news_idx, wiki,
calendar) so the model can give each kind of information its own branch and so we can measure
what each kind contributes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.store import RawData
from . import rules, text
from .align import (align_available, align_daily, available_dates, bucket_to_calendar, log_safe,
                    surprise, zscore)

log = logging.getLogger(__name__)

GROUPS = ["market", "rates", "cross", "liquidity", "macro", "forecasts", "news_idx", "wiki", "calendar"]
EXTRA_GROUPS = ["rules", "breadth"]  # only used by model variants that ask for them
# rules: classic trading rules on the index; breadth: the day-trader scanner's rules over the
# NASDAQ-100 members (today's members, so the past is computed on survivors)
ANNUAL = np.sqrt(252.0)


@dataclass
class PanelBundle:
    panel: pd.DataFrame            # features, indexed by trading day
    groups: dict[str, str]         # column -> group
    price: pd.Series               # close of the traded index on the same calendar
    cash_rate: pd.Series           # annualised cash yield in percent, as known on each day

    def columns_of(self, groups: list[str]) -> list[str]:
        return [c for c in self.panel.columns if self.groups[c] in groups]


class _Builder:
    def __init__(self, calendar: pd.DatetimeIndex):
        self.calendar = calendar
        self.columns: dict[str, pd.Series] = {}
        self.groups: dict[str, str] = {}

    def add(self, name: str, series: pd.Series | None, group: str, relative: bool = False) -> None:
        """relative=True stores a slow-moving level as its z-score against its own last three
        years, so "high" keeps meaning "high for the current regime" as decades pass."""
        if series is None:
            return
        values = series.reindex(self.calendar).astype(float).replace([np.inf, -np.inf], np.nan)
        if relative:
            values = zscore(values, window=756, min_periods=252)
        if values.notna().sum() == 0:
            log.warning("feature %s has no data and is dropped", name)
            return
        self.columns[name] = values
        self.groups[name] = group


def _daily(raw: RawData, cfg: dict, sid: str, calendar: pd.DatetimeIndex) -> pd.Series | None:
    series = raw.fred.get(sid)
    if series is None or series.empty:
        return None
    return align_daily(series, calendar, int(cfg["fred"][sid]["lag"]))


def _native(raw: RawData, sid: str) -> pd.Series | None:
    series = raw.fred.get(sid)
    return None if series is None or series.empty else series


def _publish(native: pd.Series | None, cfg: dict, sid: str, calendar: pd.DatetimeIndex,
             limit: int | None) -> pd.Series | None:
    """Align a weekly/monthly/calendar-daily transform using the series' publication delay."""
    if native is None:
        return None
    spec = cfg["fred"][sid]
    native = native.dropna()
    return align_available(native, available_dates(native.index, spec["freq"], int(spec["lag"])),
                           calendar, limit=limit)


def wiki_buckets(raw: RawData, calendar: pd.DatetimeIndex) -> pd.DataFrame | None:
    """Event counts per trading day in event time (no delay): used by the research module."""
    if raw.events is None or raw.events.empty:
        return None
    return bucket_to_calendar(text.daily_event_counts(raw.events), calendar, "sum")


def wiki_features(buckets: pd.DataFrame) -> pd.DataFrame:
    """Topic shares, tone, deaths and US focus, each as a surprise against its own last year."""
    n = buckets["n"].clip(lower=1.0)
    out = pd.DataFrame(index=buckets.index)
    for topic in text.TOPIC_PATTERNS:
        out[f"wk_{topic}"] = surprise(buckets[topic] / n, floor=0.03)
    tone = (buckets["pos"] - buckets["neg"]) / (buckets["pos"] + buckets["neg"] + 1.0)
    out["wk_tone"] = surprise(tone, floor=0.05)
    out["wk_deaths"] = surprise(np.log1p(buckets["deaths"]), floor=0.5)
    out["wk_us_share"] = surprise(buckets["us"] / n, floor=0.03)
    return out


def _calendar_features(b: _Builder, raw: RawData) -> None:
    cal = b.calendar
    days = cal.values.astype("datetime64[D]")
    g = "calendar"
    b.add("dow_mon", pd.Series((cal.dayofweek == 0).astype(float), index=cal), g)
    b.add("dow_fri", pd.Series((cal.dayofweek == 4).astype(float), index=cal), g)
    angle = 2.0 * np.pi * (cal.dayofyear.to_numpy() / 365.25)
    b.add("year_sin", pd.Series(np.sin(angle), index=cal), g)
    b.add("year_cos", pd.Series(np.cos(angle), index=cal), g)

    def days_until(event_dates: np.ndarray) -> np.ndarray:
        """Business days from each row to the next event on or after it (inf when none)."""
        event_dates = np.sort(event_dates.astype("datetime64[D]"))
        pos = np.searchsorted(event_dates, days, side="left")
        out = np.full(len(days), np.inf)
        ok = pos < len(event_dates)
        out[ok] = np.busday_count(days[ok], event_dates[pos[ok]])
        return out

    if raw.fomc is not None and len(raw.fomc):
        scheduled = raw.fomc.index[raw.fomc["scheduled"]].values
        until = days_until(scheduled)
        b.add("fomc_prox", pd.Series(np.exp(-until / 5.0), index=cal), g)
        b.add("fomc_day", pd.Series((until == 0).astype(float), index=cal), g)
        # unscheduled meetings are only known once they happen: flag the first row on/after them
        unscheduled = raw.fomc.index[~raw.fomc["scheduled"]]
        flag = pd.Series(0.0, index=cal)
        pos = cal.searchsorted(unscheduled, side="left")
        pos = pos[(pos < len(cal)) & (unscheduled >= cal[0])]
        flag.iloc[pos] = 1.0
        b.add("fomc_unsched", flag, g)

    # turn of the month: business days to the last business day of the month
    month_end = (cal + pd.offsets.BMonthEnd(0)).values
    to_month_end = np.busday_count(days, month_end.astype("datetime64[D]"))
    b.add("tom_prox", pd.Series(np.exp(-to_month_end / 3.0), index=cal), g)

    # monthly option expiry (third Friday) and quarterly "quad witching"
    months = pd.period_range(cal[0], cal[-1] + pd.Timedelta(days=100), freq="M")
    third_fridays = np.array([
        (pd.Timestamp(year=p.year, month=p.month, day=1) + pd.offsets.WeekOfMonth(week=2, weekday=4)
         if pd.Timestamp(year=p.year, month=p.month, day=1).dayofweek != 4
         else pd.Timestamp(year=p.year, month=p.month, day=15)).to_datetime64()
        for p in months])
    b.add("opex_prox", pd.Series(np.exp(-days_until(third_fridays) / 3.0), index=cal), g)
    quad = np.array([d for d in third_fridays if pd.Timestamp(d).month in (3, 6, 9, 12)])
    b.add("quad_prox", pd.Series(np.exp(-days_until(quad) / 3.0), index=cal), g)

    # US federal elections: Tuesday after the first Monday of November in even years
    def election_day(year: int) -> np.datetime64:
        first = pd.Timestamp(year=year, month=11, day=1)
        first_monday = first + pd.Timedelta(days=(7 - first.dayofweek) % 7)
        return (first_monday + pd.Timedelta(days=1)).to_datetime64()

    years = range(cal[0].year, cal[-1].year + 3)
    presidential = np.array([election_day(y) for y in years if y % 4 == 0])
    midterm = np.array([election_day(y) for y in years if y % 4 == 2])
    b.add("pres_election_prox", pd.Series(np.exp(-days_until(presidential) / 20.0), index=cal), g)
    b.add("midterm_prox", pd.Series(np.exp(-days_until(midterm) / 20.0), index=cal), g)


def build_panel(raw: RawData, cfg: dict) -> PanelBundle:
    target = cfg["project"]["target"]
    if target not in raw.fred or raw.fred[target].empty:
        raise RuntimeError(f"target series {target} is missing: run the fetch step first")
    calendar = raw.fred[target].index  # trading days = days the index has a close
    b = _Builder(calendar)

    # ------------------------------------------------------------------ market
    price = _daily(raw, cfg, target, calendar)
    logp = np.log(price)
    ret = logp.diff()
    vol10, vol20, vol60 = (ret.rolling(n).std() * ANNUAL for n in (10, 20, 60))
    g = "market"
    b.add("ret_1d", ret, g)
    b.add("ret_5d", logp.diff(5), g)
    b.add("ret_20d", logp.diff(20), g)
    b.add("ret_60d", logp.diff(60), g)
    b.add("absret_1d", ret.abs(), g)
    b.add("vol_20d", log_safe(vol20), g)
    b.add("vol_ratio", log_safe(vol10) - log_safe(vol60), g)
    b.add("drawdown", price / price.rolling(252, min_periods=60).max() - 1.0, g)
    b.add("ma_gap_50", price / price.rolling(50).mean() - 1.0, g)
    b.add("ma_gap_200", price / price.rolling(200).mean() - 1.0, g)
    b.add("up_frac_14", (ret > 0).astype(float).rolling(14).mean().where(ret.notna()), g)
    high, low = price.rolling(60).max(), price.rolling(60).min()
    b.add("range_pos_60", (price - low) / (high - low).replace(0.0, np.nan), g)
    broad = _daily(raw, cfg, "NASDAQCOM", calendar) if target != "NASDAQCOM" else None
    if broad is not None:
        b.add("rel_broad_20d", logp.diff(20) - np.log(broad).diff(20), g)
    vix = _daily(raw, cfg, "VIXCLS", calendar)
    if vix is not None:
        log_vix = np.log(vix)
        b.add("vix_log", log_vix, g)
        b.add("vix_z", zscore(log_vix), g)
        b.add("vix_chg_1d", log_vix.diff(), g)
        b.add("vix_chg_5d", log_vix.diff(5), g)
        b.add("vol_premium", vix - vol20 * 100.0, g)
        vxn = _daily(raw, cfg, "VXNCLS", calendar)
        if vxn is not None:
            b.add("vxn_vix", np.log(vxn) - log_vix, g)

    # ------------------------------------------------------------------ rates and credit
    g = "rates"
    y10, y2, y3m, y30 = (_daily(raw, cfg, s, calendar) for s in ("DGS10", "DGS2", "DGS3MO", "DGS30"))
    fed_funds, real10 = _daily(raw, cfg, "DFF", calendar), _daily(raw, cfg, "DFII10", calendar)
    breakeven, forward = _daily(raw, cfg, "T10YIE", calendar), _daily(raw, cfg, "T5YIFR", calendar)
    if y10 is not None:
        b.add("y10_z", zscore(y10), g)
        b.add("y10_chg_1d", y10.diff(), g)
        b.add("y10_chg_20d", y10.diff(20), g)
    if y2 is not None:
        b.add("y2_chg_1d", y2.diff(), g)
        b.add("y2_chg_20d", y2.diff(20), g)
    if y10 is not None and y2 is not None:
        b.add("slope_10y2y", y10 - y2, g)
        b.add("slope_chg_20d", (y10 - y2).diff(20), g)
    if y10 is not None and y3m is not None:
        b.add("slope_10y3m", y10 - y3m, g)
    if y30 is not None and y10 is not None:
        b.add("curve_30y10y", y30 - y10, g, relative=True)
    if real10 is not None:
        b.add("real10", real10, g, relative=True)
        b.add("real10_chg_20d", real10.diff(20), g)
    if breakeven is not None:
        b.add("breakeven10", breakeven, g, relative=True)
        b.add("breakeven10_chg_20d", breakeven.diff(20), g)
    if forward is not None:
        b.add("fwd5y5y_chg_20d", forward.diff(20), g)
    if y2 is not None and fed_funds is not None:
        b.add("policy_gap", y2 - fed_funds, g, relative=True)
        b.add("ff_chg_60d", fed_funds.diff(60), g)
    baa, aaa = _daily(raw, cfg, "BAA10Y", calendar), _daily(raw, cfg, "AAA10Y", calendar)
    if baa is not None:
        b.add("baa_spread", baa, g, relative=True)
        b.add("baa_chg_5d", baa.diff(5), g)
        b.add("baa_chg_20d", baa.diff(20), g)
        if aaa is not None:
            b.add("quality_spread", baa - aaa, g, relative=True)

    # ------------------------------------------------------------------ currencies and oil
    g = "cross"
    for sid, name in (("DEXUSEU", "eur"), ("DEXJPUS", "jpy")):
        fx = _daily(raw, cfg, sid, calendar)
        if fx is not None:
            b.add(f"{name}_ret_5d", np.log(fx).diff(5), g)
            b.add(f"{name}_ret_20d", np.log(fx).diff(20), g)
    oil = _daily(raw, cfg, "DCOILWTICO", calendar)
    if oil is not None:
        log_oil = np.log(oil.clip(lower=1.0))  # WTI settled below zero once (April 2020)
        b.add("oil_ret_5d", log_oil.diff(5), g)
        b.add("oil_ret_20d", log_oil.diff(20), g)
        b.add("oil_vol_20d", log_oil.diff().rolling(20).std(), g)

    # ------------------------------------------------------------------ liquidity / conditions
    g = "liquidity"
    nfci = _native(raw, "NFCI")
    if nfci is not None:
        b.add("nfci", _publish(nfci, cfg, "NFCI", calendar, 15), g)
        b.add("nfci_chg_4w", _publish(nfci.diff(4), cfg, "NFCI", calendar, 15), g)
    b.add("stress", _publish(_native(raw, "STLFSI4"), cfg, "STLFSI4", calendar, 15), g)
    balance = _native(raw, "WALCL")
    if balance is not None:
        b.add("fed_assets_13w", _publish(np.log(balance).diff(13), cfg, "WALCL", calendar, 15), g)
    claims = _native(raw, "ICSA")
    if claims is not None:
        deviation = np.log(claims.rolling(4).mean()) - np.log(claims.rolling(52).mean())
        b.add("claims_dev", _publish(deviation, cfg, "ICSA", calendar, 15), g)

    # ------------------------------------------------------------------ monthly macro
    g = "macro"
    unrate = _native(raw, "UNRATE")
    if unrate is not None:
        b.add("unrate_chg_3m", _publish(unrate.diff(3), cfg, "UNRATE", calendar, 60), g)
    b.add("sahm", _publish(_native(raw, "SAHMREALTIME"), cfg, "SAHMREALTIME", calendar, 60), g)
    cpi = _native(raw, "CPIAUCNS")
    if cpi is not None:
        yoy = cpi.pct_change(12) * 100.0
        b.add("cpi_yoy", _publish(yoy, cfg, "CPIAUCNS", calendar, 60), g, relative=True)
        b.add("cpi_yoy_chg_3m", _publish(yoy.diff(3), cfg, "CPIAUCNS", calendar, 60), g)
    for sid, name in (("INDPRO", "indpro_yoy"), ("PERMIT", "permit_yoy")):
        native = _native(raw, sid)
        if native is not None:
            b.add(name, _publish(native.pct_change(12) * 100.0, cfg, sid, calendar, 60), g)
    sentiment = _native(raw, "UMCSENT")
    if sentiment is not None:
        b.add("consumer_sent", _publish(sentiment, cfg, "UMCSENT", calendar, 60), g, relative=True)
        b.add("consumer_sent_chg_3m", _publish(sentiment.diff(3), cfg, "UMCSENT", calendar, 60), g)

    # ------------------------------------------------------------------ forecasts / expectations
    g = "forecasts"
    if raw.spf is not None and len(raw.spf):
        spf = align_available(raw.spf, raw.spf.index, calendar, limit=110)
        for col in spf.columns:
            b.add(col, spf[col], g, relative=(col == "spf_cpi"))
    mich = _native(raw, "MICH")
    if mich is not None:
        b.add("infl_exp_consumer", _publish(mich, cfg, "MICH", calendar, 60), g, relative=True)
        b.add("infl_exp_consumer_chg_3m", _publish(mich.diff(3), cfg, "MICH", calendar, 60), g)
    b.add("infl_exp_1y", _publish(_native(raw, "EXPINF1YR"), cfg, "EXPINF1YR", calendar, 60), g, relative=True)
    b.add("infl_exp_10y", _publish(_native(raw, "EXPINF10YR"), cfg, "EXPINF10YR", calendar, 60), g, relative=True)

    # ------------------------------------------------------------------ newspaper-based indices
    g = "news_idx"
    src = cfg["sources"]
    if raw.sentiment is not None and len(raw.sentiment):
        s = raw.sentiment
        avail = available_dates(s.index, "c", int(src["news_sentiment"]["lag_days"]))
        b.add("news_sent", align_available(s, avail, calendar, limit=15), g)
        b.add("news_sent_chg_7d", align_available(s.diff(7), avail, calendar, limit=15), g)
        b.add("news_sent_chg_28d", align_available(s.diff(28), avail, calendar, limit=15), g)
    for sid, name in (("USEPUINDXD", "policy_unc"), ("WLEMUINDXD", "equity_unc")):
        native = _native(raw, sid)
        if native is not None:
            week = np.log(native.rolling(7, min_periods=4).mean().clip(lower=1.0))
            quarter = np.log(native.rolling(90, min_periods=30).mean().clip(lower=1.0))
            b.add(f"{name}_7d", _publish(week, cfg, sid, calendar, 10), g, relative=True)
            b.add(f"{name}_ratio", _publish(week - quarter, cfg, sid, calendar, 10), g)
    if raw.gpr is not None and len(raw.gpr):
        avail = available_dates(raw.gpr.index, "c", int(src["gpr"]["lag_days"]))
        for col in ("gpr", "gpr_act", "gpr_threat"):
            month = np.log(raw.gpr[col].rolling(30, min_periods=10).mean().clip(lower=1.0))
            b.add(f"{col}_30d", align_available(month, avail, calendar, limit=40), g, relative=True)

    # ------------------------------------------------------------------ Wikipedia events
    buckets = wiki_buckets(raw, calendar)
    if buckets is not None:
        delay = int(src["wiki"].get("lag_rows", 1))  # give editors a day to fill in the page
        features = wiki_features(buckets).shift(delay)
        for col in features.columns:
            b.add(col, features[col], "wiki")

    # ------------------------------------------------------------------ calendar
    _calendar_features(b, raw)

    # ------------------------------------------------------------------ market breadth (day-trader scanner rules)
    if raw.stock_closes is not None and len(raw.stock_closes):
        from ..stocks.setups import breadth as stock_breadth
        br = stock_breadth(raw.stock_closes)
        for col in [c for c in br.columns if c != "br_members"]:
            b.add(col, align_daily(br[col], calendar, 0, limit=3), "breadth")

    # ------------------------------------------------------------------ classic trading rules
    signals = rules.rule_signals(price)
    for col in signals.columns:
        b.add(f"rule_{col}", signals[col], "rules")
    macd = rules.macd_lines(price)  # the rule above only says in/out; these carry the strength
    b.add("macd_line", macd["macd"], "rules")
    b.add("macd_hist", macd["macd_hist"], "rules")
    b.add("macd_hist_chg_5d", macd["macd_hist"].diff(5), "rules")

    panel = pd.DataFrame(b.columns, index=calendar)
    start = pd.Timestamp(cfg["project"]["start"])
    keep = panel.index >= start
    panel = panel[keep]
    cash_sid = cfg["trading"]["cash_rate_series"]
    cash = _daily(raw, cfg, cash_sid, calendar)
    cash = pd.Series(0.0, index=calendar) if cash is None else cash.ffill().fillna(0.0)
    return PanelBundle(panel=panel, groups=dict(b.groups), price=price[keep], cash_rate=cash[keep])
