"""Virtual-money bookkeeping and the equivalence of backtest and forward paper account."""

import numpy as np
import pandas as pd

from marketlab.trading import metrics, paper
from marketlab.trading.backtest import simulate
from marketlab.trading.broker import PaperBroker
from marketlab.trading.strategy import target_weights

TCFG = {"initial_cash": 100000.0, "cost_bps": 5.0, "max_weight": 1.0, "allow_short": False,
        "rebalance_band": 0.10, "signal_smooth": 1, "sizing": "relative", "gain": 1.0}


def test_broker_pays_costs_and_lands_on_target():
    broker = PaperBroker(cash=100000.0, cost_bps=10.0)
    trade = broker.rebalance_to(1.0, price=50.0, date="2020-01-02")
    assert trade["side"] == "BUY"
    assert np.isclose(broker.equity(50.0), 100000.0 - trade["cost"])
    assert np.isclose(broker.weight(50.0), 1.0, atol=1e-9)
    assert abs(broker.cash) < 1e-6                      # fully invested, cost included
    broker.rebalance_to(0.0, price=55.0, date="2020-01-03")
    assert abs(broker.units) < 1e-12
    gross = (100000.0 - trade["cost"]) * 55.0 / 50.0
    assert np.isclose(broker.cash, gross * (1 - 10.0 / 1e4))
    assert np.isclose(broker.costs_paid, trade["cost"] + gross * 10.0 / 1e4)


def test_interest_accrues_on_cash_only():
    broker = PaperBroker(cash=100000.0)
    earned = broker.accrue_interest(annual_rate_pct=3.65, days=10)
    assert np.isclose(earned, 100.0) and np.isclose(broker.cash, 100100.0)
    assert broker.accrue_interest(3.65, 0) == 0.0


def test_buy_and_hold_tracks_the_index_and_fills_one_day_late():
    dates = pd.bdate_range("2020-01-01", periods=30)
    price = pd.Series(np.linspace(100, 130, 30), index=dates)
    cash = pd.Series(0.0, index=dates)
    equity, trades = simulate(price, pd.Series(1.0, index=dates), cash, {**TCFG, "cost_bps": 0.0}, 1)
    assert len(trades) == 1 and trades.iloc[0]["date"] == str(dates[1].date())  # decided day 0, filled day 1
    assert np.isclose(equity["equity"].iloc[0], 100000.0)
    assert np.isclose(equity["equity"].iloc[-1], 100000.0 * price.iloc[-1] / price.iloc[1])


def test_rebalance_band_suppresses_small_trades():
    dates = pd.bdate_range("2020-01-01", periods=10)
    price = pd.Series(100.0, index=dates)
    targets = pd.Series([0.5, 0.55, 0.52, 0.58, 0.5, 0.75, 0.75, 0.0, 0.0, 0.0], index=dates)
    _, trades = simulate(price, targets, pd.Series(0.0, index=dates), TCFG, 1)
    assert [t for t in trades["weight_after"].round(2)] == [0.5, 0.75, 0.0]


def test_relative_sizing():
    index = pd.bdate_range("2020-01-01", periods=5)
    pred = pd.DataFrame({"p_5": [0.60, 0.55, 0.50, 0.65, 0.40], "ref_mean_5": 0.60, "ref_std_5": 0.05,
                         "base_p_5": 0.56}, index=index)
    weights = target_weights(pred, TCFG, 5)
    # at its usual level -> invested; one sd below -> half; two sd below -> out; never above the cap
    assert weights.round(6).tolist() == [1.0, 0.5, 0.0, 1.0, 0.0]
    missing = pred.assign(ref_std_5=np.nan)
    assert (target_weights(missing, TCFG, 5) == 1.0).all()   # no yardstick yet: stay invested


def test_paper_account_reproduces_the_backtest(bundle, cfg):
    """Stepping the forward account day by day must give exactly the backtest's equity."""
    dates = bundle.price.index[-120:]
    rng = np.random.default_rng(5)
    pred = pd.DataFrame({"p_5": rng.uniform(0.4, 0.7, len(dates)), "ref_mean_5": 0.55, "ref_std_5": 0.06,
                         "base_p_5": 0.56}, index=dates)
    horizon = int(cfg["model"]["trade_horizon"])
    targets = target_weights(pred, cfg["trading"], horizon)
    equity, trades = simulate(bundle.price, targets, bundle.cash_rate, cfg["trading"], 1)

    class Cut:  # a bundle that only knows the first `n` days, as the account would have seen it
        def __init__(self, n):
            self.price, self.cash_rate = bundle.price.loc[:dates[n - 1]], bundle.cash_rate.loc[:dates[n - 1]]

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        first = Cut(1)
        first.price, first.cash_rate = first.price.loc[dates[0]:], first.cash_rate.loc[dates[0]:]
        state = paper.init_account(first, pred.iloc[:1], cfg, Path(tmp))
        for n in (2, 40, 41, 120):      # irregular catch-ups, like a user who skips days
            state = paper.advance(Cut(n), pred.iloc[:n], cfg, state)
        info = paper.summary(state)
    assert np.isclose(info["equity"], round(float(equity["equity"].iloc[-1]), 2))
    assert info["trades"] == len(trades)


def test_metrics_basics():
    assert np.isclose(metrics.auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])), 0.75)
    assert np.isclose(metrics.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    assert np.isclose(metrics.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
    index = pd.bdate_range("2020-01-01", periods=253)
    equity = pd.Series(100.0 * 1.0004 ** np.arange(253), index=index)
    perf = metrics.performance(equity)
    assert np.isclose(perf["total_return"], 1.0004 ** 252 - 1)
    assert perf["max_drawdown"] == 0.0


def test_classic_rules_are_bounded_and_behave_on_simple_paths():
    from marketlab.features.rules import RULES, rule_signals
    dates = pd.bdate_range("2015-01-01", periods=600)
    rising = pd.Series(np.exp(np.linspace(0, 1, 600)), index=dates)
    up = rule_signals(rising)
    assert list(up.columns) == list(RULES)
    assert up.min().min() >= 0.0 and up.max().max() <= 1.0
    last = up.iloc[-1]
    assert last["trend_200d"] == 1 and last["ma_cross_50_200"] == 1 and last["momentum_12m"] == 1
    assert last["breakout_55_20"] == 1 and last["bollinger_reversion"] == 0
    down = rule_signals(rising[::-1].set_axis(dates)).iloc[-1]
    assert down["trend_200d"] == 0 and down["momentum_12m"] == 0 and down["breakout_55_20"] == 0
    # causal: appending future days must not change earlier signals
    np.testing.assert_allclose(rule_signals(rising.iloc[:400]).to_numpy(), up.iloc[:400].to_numpy(), equal_nan=True)
