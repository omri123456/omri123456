"""Turn event text into daily numbers with transparent keyword lexicons.

The same lexicons are applied to all years, so the meaning of a feature does not change when
Wikipedia changed its page layout (category headings only exist from about 2010).
Every topic is a plain list of phrases: read it, edit it, add your own.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

TOPICS: dict[str, list[str]] = {
    "war": [
        r"war", r"wars", r"airstrikes?", r"air strikes?", r"troops", r"invasion", r"invades?",
        r"military offensive", r"offensive", r"shelling", r"missiles?", r"rebels?", r"insurgents?",
        r"ceasefire", r"cease-fire", r"armed forces", r"army", r"soldiers?", r"militia", r"drone strikes?",
        r"bombardment", r"frontline", r"artillery", r"combat", r"warplanes?", r"naval blockade",
    ],
    "terror": [
        r"terror\w*", r"suicide bomb\w*", r"car bomb\w*", r"bombings?", r"hostages?", r"militants?",
        r"al-qaeda", r"al qaeda", r"isis", r"isil", r"islamic state", r"taliban", r"boko haram",
        r"hijack\w*", r"gunmen", r"mass shooting",
    ],
    "disaster": [
        r"earthquakes?", r"hurricanes?", r"typhoons?", r"cyclones?", r"tsunamis?", r"floods?", r"flooding",
        r"wildfires?", r"bushfires?", r"volcano\w*", r"eruption", r"tornado\w*", r"landslides?", r"mudslides?",
        r"plane crash\w*", r"crashes", r"derail\w*", r"explosion", r"drought", r"heat ?wave", r"blizzard",
        r"shipwreck", r"capsizes?", r"collapses", r"state of emergency",
    ],
    "health": [
        r"pandemic", r"epidemic", r"outbreak", r"virus", r"coronavirus", r"covid\S*", r"ebola", r"h1n1",
        r"sars", r"mers", r"influenza", r"swine flu", r"bird flu", r"avian flu", r"quarantine\w*",
        r"lockdowns?", r"vaccin\w*", r"world health organization", r"infections?", r"cholera", r"zika",
    ],
    "economy": [
        r"economy", r"economic", r"recession", r"gdp", r"gross domestic product", r"inflation",
        r"unemployment", r"jobless", r"jobs report", r"consumer prices?", r"deficit", r"austerity",
        r"stimulus", r"budget", r"national debt", r"public debt", r"imf", r"international monetary fund",
        r"world bank", r"g20", r"g7", r"g8",
    ],
    "central_bank": [
        r"federal reserve", r"the fed", r"interest rates?", r"central banks?", r"european central bank",
        r"ecb", r"bank of england", r"bank of japan", r"people's bank of china", r"rate cut\w*",
        r"rate hike\w*", r"raises? (?:its )?(?:key |benchmark )?(?:interest )?rates?",
        r"cuts? (?:its )?(?:key |benchmark )?(?:interest )?rates?", r"quantitative easing",
        r"monetary policy", r"fomc", r"basis points?",
    ],
    "fin_crisis": [
        r"bankrupt\w*", r"bailouts?", r"bail-outs?", r"defaults? on", r"sovereign default", r"debt crisis",
        r"financial crisis", r"credit crunch", r"credit crisis", r"subprime", r"rescue package",
        r"nationali[sz]\w+", r"insolven\w+", r"liquidity", r"downgrad\w+", r"credit rating",
        r"bank run", r"banking crisis", r"chapter 11", r"receivership", r"write-?downs?", r"meltdown",
        r"emergency loans?", r"contagion",
    ],
    "markets": [
        r"stock markets?", r"stock exchange", r"stocks", r"shares", r"wall street", r"dow jones",
        r"s&p 500", r"s&p", r"nasdaq", r"ftse", r"nikkei", r"hang seng", r"dax", r"record high",
        r"all-time high", r"sell-?off", r"bear market", r"bull market", r"market crash", r"trading halt\w*",
        r"circuit breaker", r"bond yields?", r"treasury yields?", r"cryptocurrenc\w+", r"bitcoin",
    ],
    "corporate": [
        r"merger", r"acquisitions?", r"acquires?", r"takeover", r"buyout", r"ipo", r"initial public offering",
        r"layoffs?", r"lay off", r"job cuts", r"cut \S+ jobs", r"earnings", r"quarterly profit", r"ceo",
        r"chief executive", r"antitrust", r"shareholders?", r"corporation", r"automaker", r"airline",
    ],
    "energy": [
        r"oil prices?", r"crude oil", r"crude", r"opec", r"oil", r"natural gas", r"gas pipeline", r"pipeline",
        r"gasoline", r"barrel", r"oil spill", r"refinery", r"nuclear plant", r"power outage", r"blackout",
        r"energy crisis",
    ],
    "trade": [
        r"tariffs?", r"trade war", r"sanctions?", r"embargo", r"trade deal", r"trade agreement",
        r"free trade", r"world trade organization", r"wto", r"export ban", r"import ban", r"trade talks",
        r"trade dispute", r"protectionis\w+",
    ],
    "politics": [
        r"elections?", r"votes?", r"voters?", r"referendum", r"president", r"prime minister", r"parliament",
        r"coup", r"impeach\w*", r"resigns?", r"resignation", r"cabinet", r"coalition", r"inaugurat\w+",
        r"ballot", r"sworn in",
    ],
    "us_policy": [
        r"congress", r"senate", r"house of representatives", r"white house", r"u\.s\. president",
        r"debt ceiling", r"government shutdown", r"supreme court", r"executive order", r"tax cuts?",
        r"treasury secretary", r"democrats?", r"republicans?", r"filibuster",
    ],
    "geopolitics": [
        r"summit", r"treaty", r"nuclear", r"north korea", r"iran", r"russia", r"china", r"nato",
        r"united nations", r"security council", r"diplomat\w*", r"ambassador", r"annex\w+", r"sovereignty",
        r"border dispute", r"taiwan", r"ukraine", r"middle east", r"israel", r"gaza",
    ],
    "unrest": [
        r"protests?", r"protesters", r"riots?", r"rioting", r"demonstrat\w+", r"general strike", r"unrest",
        r"clashes", r"uprising", r"revolution", r"curfew", r"martial law", r"crackdown", r"tear gas",
    ],
    "crime_law": [
        r"shooting", r"murder\w*", r"arrest\w*", r"trial", r"sentenced", r"convicted", r"indict\w+", r"fraud",
        r"scandal", r"corruption", r"lawsuit", r"court rules", r"guilty", r"money laundering", r"ponzi",
        r"insider trading",
    ],
    "tech": [
        r"cyber ?attacks?", r"hack\w*", r"data breach", r"ransomware", r"artificial intelligence",
        r"semiconductors?", r"chips?", r"microsoft", r"apple", r"google", r"amazon", r"facebook", r"meta",
        r"nvidia", r"tesla", r"spacex", r"nasa", r"satellite", r"internet outage", r"social media",
    ],
}

NEGATIVE = [
    r"crisis", r"crises", r"collapse\w*", r"fears?", r"plunge\w*", r"plummet\w*", r"crash\w*", r"recession",
    r"defaults?", r"fraud", r"scandal", r"kill\w*", r"dead", r"deaths?", r"dies?", r"died", r"attack\w*",
    r"bomb\w*", r"disaster", r"emergency", r"sanctions?", r"shutdown", r"downgrad\w+", r"loss(?:es)?",
    r"layoffs?", r"riots?", r"violence", r"violent", r"threat\w*", r"warn\w*", r"fails?", r"failed",
    r"failure", r"bankrupt\w*", r"injur\w+", r"wound\w+", r"destroy\w*", r"damage\w*", r"victims?",
    r"evacuat\w+", r"suspend\w*", r"ban(?:s|ned)?", r"arrest\w*", r"conflict", r"tensions?", r"slump\w*",
    r"panic", r"turmoil", r"worst", r"decline\w*", r"falls?", r"fell", r"drops?", r"dropped", r"cuts?",
    r"shortages?", r"strikes?", r"outbreak", r"hostages?", r"casualties", r"missing", r"guilty",
]

POSITIVE = [
    r"agreement", r"agrees?", r"deal", r"peace", r"ceasefire", r"recover\w*", r"growth", r"record high",
    r"wins?", r"won", r"approv\w+", r"signs?", r"signed", r"launch\w*", r"rescue\w*", r"breakthrough",
    r"surplus", r"rally", r"rallies", r"gains?", r"rebound\w*", r"celebrat\w+", r"reopen\w*", r"released?",
    r"elected", r"success\w*", r"lifts?", r"lifted", r"ends? (?:the|a|its)", r"restor\w+", r"relief",
    r"aid", r"cure", r"discover\w*", r"upgrade\w*", r"boost\w*", r"rises?", r"rose", r"surges?", r"surged",
    r"improv\w+", r"resolv\w+", r"reach(?:es)? (?:a|an) (?:deal|agreement)", r"treaty", r"awarded",
]

# Case-sensitive on purpose: "US" must not match the pronoun "us".
US_PATTERN = re.compile(
    r"\b(?:United States|U\.S\.|US|USA|American|Americans|Washington|White House|Congress|Senate|"
    r"Wall Street|Federal Reserve|Pentagon|New York|California|Texas|Florida)\b")

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "dozen": 12, "dozens": 24,
                 "hundreds": 200, "thousands": 2000}
_NUM = r"(\d[\d,]*|" + "|".join(_NUMBER_WORDS) + r")"
_QUALIFIER = r"(?:at least |more than |over |around |about |nearly |up to |some |as many as |an estimated )?"
_DEATH_AFTER = re.compile(
    r"(?:kill(?:s|ed|ing)?|leav(?:es|ing)|claim(?:s|ed|ing) the lives of|death toll[\w\s]{0,30}?"
    r"(?:to|at|reaches|passes|exceeds|tops))\s+" + _QUALIFIER + _NUM + r"\b", re.I)
_DEATH_BEFORE = re.compile(
    _NUM + r"\s+(?:\w+\s+){0,3}?(?:are |were |have been |is |was )?"
    r"(?:killed|dead|died|die|deaths|fatalities|drown(?:ed)?|perish(?:ed)?)\b", re.I)


def _compile(phrases: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(phrases) + r")\b", re.I)


TOPIC_PATTERNS = {name: _compile(phrases) for name, phrases in TOPICS.items()}
NEGATIVE_PATTERN = _compile(NEGATIVE)
POSITIVE_PATTERN = _compile(POSITIVE)


def _to_number(token: str) -> float:
    token = token.lower()
    if token in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[token])
    has_comma = "," in token
    try:
        value = float(token.replace(",", ""))
    except ValueError:
        return 0.0
    if not has_comma and 1900 <= value <= 2100:  # almost certainly a year, not a death count
        return 0.0
    return value if value <= 1e6 else 0.0


def death_count(text: str) -> float:
    """Largest death count mentioned in one event text (0 when none is found)."""
    best = 0.0
    for pattern in (_DEATH_AFTER, _DEATH_BEFORE):
        for match in pattern.finditer(text):
            best = max(best, _to_number(match.group(1)))
    return best


def score_events(events: pd.DataFrame) -> pd.DataFrame:
    """Per-event scores: one 0/1 column per topic plus tone counts, deaths and US focus."""
    text = events["text"].astype(str)
    out = pd.DataFrame({"date": events["date"].to_numpy()})
    for name, pattern in TOPIC_PATTERNS.items():
        out[name] = text.str.contains(pattern).to_numpy(dtype=float)
    out["neg"] = text.str.count(NEGATIVE_PATTERN).to_numpy(dtype=float)
    out["pos"] = text.str.count(POSITIVE_PATTERN).to_numpy(dtype=float)
    out["us"] = text.str.contains(US_PATTERN).to_numpy(dtype=float)
    out["deaths"] = np.fromiter((death_count(t) for t in text), dtype=float, count=len(text))
    out["n"] = 1.0
    return out


def daily_event_counts(events: pd.DataFrame) -> pd.DataFrame:
    """Sum the per-event scores by calendar day (weekends included, days without events = 0)."""
    columns = list(TOPIC_PATTERNS) + ["neg", "pos", "us", "deaths", "n"]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns, dtype=float)
    scored = score_events(events)
    daily = scored.groupby("date")[columns].sum()
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(full_index, fill_value=0.0)
