"""Paper broker: one account, one instrument, virtual cash.

The same object runs the historical backtest and the forward paper account, so both follow
identical rules: orders fill at the given close, every trade pays a cost, idle cash earns the
short rate, and nothing is created or lost outside those three mechanisms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class PaperBroker:
    cash: float
    cost_bps: float = 5.0          # one-way cost in basis points of traded value
    borrow_spread: float = 1.0     # extra annual percent paid on negative cash (leverage)
    units: float = 0.0             # units of the index held (can be fractional or negative)
    last_price: float | None = None
    last_date: str | None = None
    trades: list[dict] = field(default_factory=list)
    costs_paid: float = 0.0
    interest_earned: float = 0.0

    # ------------------------------------------------------------------ valuation
    def equity(self, price: float | None = None) -> float:
        price = self.last_price if price is None else price
        return self.cash + self.units * (price or 0.0)

    def weight(self, price: float | None = None) -> float:
        price = self.last_price if price is None else price
        total = self.equity(price)
        return 0.0 if total <= 0 or price is None else self.units * price / total

    # ------------------------------------------------------------------ daily bookkeeping
    def accrue_interest(self, annual_rate_pct: float, days: int) -> float:
        """Interest on cash for `days` calendar days at an annual percentage rate."""
        if days <= 0:
            return 0.0
        rate = annual_rate_pct if self.cash >= 0 else annual_rate_pct + self.borrow_spread
        interest = self.cash * (rate / 100.0) * days / 365.0
        self.cash += interest
        self.interest_earned += interest
        return interest

    def mark(self, date: str, price: float) -> float:
        self.last_date, self.last_price = date, float(price)
        return self.equity()

    # ------------------------------------------------------------------ trading
    def rebalance_to(self, target_weight: float, price: float, date: str, note: str = "") -> dict | None:
        """Trade at `price` so the position is target_weight of equity after paying the cost."""
        price = float(price)
        k = self.cost_bps / 1e4
        equity_before = self.equity(price)
        if equity_before <= 0:
            return None
        current_value = self.units * price
        gap = target_weight * equity_before - current_value
        # the cost comes out of equity, so solve for the trade that lands exactly on target
        trade_value = gap / (1.0 + target_weight * k) if gap >= 0 else gap / (1.0 - target_weight * k)
        if abs(trade_value) < 1e-9:
            return None
        cost = abs(trade_value) * k
        self.units += trade_value / price
        self.cash -= trade_value + cost
        self.costs_paid += cost
        trade = {"date": date, "side": "BUY" if trade_value > 0 else "SELL", "price": price,
                 "value": round(trade_value, 2), "units": trade_value / price, "cost": round(cost, 2),
                 "weight_after": round(self.weight(price), 4), "equity_after": round(self.equity(price), 2),
                 "note": note}
        self.trades.append(trade)
        return trade

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PaperBroker":
        return cls(**data)
