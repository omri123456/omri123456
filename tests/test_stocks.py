"""The bridge to the day-trader skill: member parsing, the scanner rules and breadth."""

import numpy as np
import pandas as pd

from marketlab.stocks import combine, history, setups

MEMBERS = """{| class="wikitable sortable" id="constituents"
|-
! Ticker !! Company !! ISIN !! Industry !! Subsector
|-
| ADBE || [[Adobe Inc.]] || {{ISIN|sl=n|US00724F1012}} || Technology || Software
|-
| GOOGL || [[Alphabet Inc.]] (Class A) || x || Technology || Software
|-
| AMZN || [[Amazon (company)|Amazon]] || x || Consumer Discretionary || Retail
|}"""


def test_member_table_parsing():
    m = history.parse_members(MEMBERS)
    assert m["symbol"].tolist() == ["ADBE", "GOOGL", "AMZN"]
    assert m.loc[2, "company"] == "Amazon" and m.loc[0, "industry"] == "Technology"


def _closes():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2015-01-01", periods=400)
    up = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.002, 0.01, 400))), index=idx)
    down = pd.Series(100 * np.exp(np.cumsum(rng.normal(-0.004, 0.01, 400))), index=idx)
    late = up.copy(); late.iloc[:300] = np.nan          # listed later: no rules before 30 closes
    return pd.DataFrame({"UP": up, "DOWN": down, "LATE": late})


def test_rules_are_causal_and_need_history():
    c = _closes()
    flags = setups.setup_flags(c)
    cut = setups.setup_flags(c.iloc[:250])
    for k in setups.SETUPS:
        assert (flags[k].iloc[:250] == cut[k]).all().all(), k   # appending days changes nothing earlier
        assert not flags[k]["LATE"].iloc[:329].any(), k          # fewer than 30 closes: no signal
    assert flags["oversold"]["DOWN"].sum() > flags["oversold"]["UP"].sum()


def test_breadth_shares_and_regimes():
    c = _closes()
    br = setups.breadth(c, min_members=2)
    assert br["br_above_ma200"].dropna().between(0, 1).all()
    assert br["br_members"].iloc[-1] == 3 and br["br_members"].iloc[100] == 2
    score = pd.Series([-1.0, 0.0, 0.7, np.nan])
    assert combine.regime_of(score).tolist()[:3] == ["pessimistic", "neutral", "optimistic"]
    assert pd.isna(combine.regime_of(score).iloc[3])
