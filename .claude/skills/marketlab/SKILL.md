---
name: marketlab
description: >-
  MarketLab's daily view of the NASDAQ-100: the CNN's decision for the next close (buy, add,
  reduce, exit or hold), the target exposure, the model's probability of a rise against its own
  usual level, market breadth from the day-trader scanner, what followed the most similar past
  days, and the virtual paper account. Use when the user asks "what does the model say today",
  "should I be in the market", "morning market brief", "how is the paper account doing", "how
  broad is the selloff", or "run the daily update". Reads files from a checkout of this
  repository; running the model needs Python with PyTorch.
license: MIT
compatibility: Claude Code, OpenJarvis and other agentskills.io readers. Reading needs only a checkout; the daily update needs Python 3.11+ and the packages in requirements.txt.
metadata:
  repository: https://github.com/omri123456/omri123456
  branch: claude/market-prediction-cnn-5aj5ok
---

# marketlab

MarketLab is a research system, not an adviser. Its CNN reads 60 trading days of about 115
point-in-time series (prices and volatility, rates and credit, currencies and oil, liquidity,
macro data, economists' forecasts, newspaper indices, Wikipedia world events, Fed meetings) and
gives the probability that the NASDAQ-100 rises over the next 1, 5 and 20 trading days. The
strategy holds the index and cuts exposure when the model is more pessimistic than it has been
over the previous year. In the clean test (2019 on) it returned about the same as holding the
index with a smaller worst fall; it has not been shown to beat buy and hold.

## Answer from the files (no Python needed)

Pull the branch first (`git pull`), then read:

| file | what it holds |
| --- | --- |
| `state/latest_signal.json` | today's decision: `decision` (buy, add, reduce, exit, hold), `decision_he`, `weight_now`, `weight_target`, `p_up`, `p_usual`, `score`, `account`, `breadth` |
| `state/signals.csv` | one row per day of past decisions |
| `state/breadth.csv` | share of ~1,500 liquid US stocks in each scanner setup, per day |
| `state/paper_account.json` | the virtual account: `history`, `broker.trades`, `pending` order |
| `reports/dashboard/outlook.json` | what the index did 5 and 20 days after the 60 most similar past days |
| `reports/dashboard/scorecard.json` | backtest numbers for the model and its benchmarks |
| `runs/latest/strategies.csv`, `runs/latest/prediction_quality.csv` | full backtest tables |

A morning brief is three lines: the decision and its date, exposure now and target with the
probability against the usual level, and the account's value with the market's breadth. Say
the date of the data. If `latest_signal.json` is older than two trading days, say the daily
update did not run.

## Run the daily update (needs Python)

```bash
python3 -m pip install -r requirements.txt
python3 -m marketlab daily      # fetch data, step the paper account, breadth scan, signal, report, dashboard data
python3 -m marketlab signal     # print today's decision again
python3 -m marketlab status     # what is on disk and how fresh it is
```

## Rules

- Report what the model says; never turn it into a personal recommendation, and never promise a
  return. Say it is an experiment when someone asks what to do with real money.
- The model speaks only about the NASDAQ-100 as a whole. Scanner hits on single stocks matched a
  chart rule; they are not picks.
- The outlook from similar days is history, not a forecast: tested walk-forward, it did not tell
  good days from bad ones better than the base rate.
- The paper account is virtual money. Nothing here places real orders.
