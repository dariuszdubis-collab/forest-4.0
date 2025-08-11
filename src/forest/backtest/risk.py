from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class RiskManager:
    initial_capital: float
    risk_per_trade: float = 0.01
    max_drawdown: float = 0.3
    fee_perc: float = 0.0005
    slippage_perc: float = 0.0

    # state
    _cash: float = field(init=False)
    _position: float = field(default=0.0, init=False)  # qty
    _avg_price: float = field(default=0.0, init=False)
    _equity_curve: List[float] = field(default_factory=list, init=False)
    _peak: float = field(init=False)

    def __post_init__(self):
        self._cash = self.initial_capital
        self._peak = self.initial_capital
        self._equity_curve = [self.initial_capital]

    @property
    def equity(self) -> float:
        return self._equity_curve[-1]

    def record_mark_to_market(self, price: float):
        equity = self._cash + self._position * price
        self._equity_curve.append(equity)
        self._peak = max(self._peak, equity)

    def exceeded_max_dd(self) -> bool:
        dd = 0.0 if self._peak == 0 else 1.0 - (self.equity / self._peak)
        return dd > self.max_drawdown

    def position_size(self, price: float, atr: float, atr_multiple: float) -> float:
        if atr <= 0:
            return 0.0
        risk_amount = self.equity * self.risk_per_trade
        unit_risk = atr * atr_multiple
        qty = risk_amount / unit_risk
        # ensure we can afford it (very rough check)
        max_qty_by_cash = max(0.0, self._cash / (price * (1 + self.fee_perc + self.slippage_perc)))
        return max(0.0, min(qty, max_qty_by_cash))

    def _position_cost(self, price: float, qty: float) -> float:
        gross = price * qty
        return gross * (self.fee_perc + self.slippage_perc)

    def buy(self, price: float, qty: float):
        if qty <= 0:
            return
        cost = price * qty + self._position_cost(price, qty)
        if cost > self._cash:
            return  # cannot afford
        # average price update
        total_cost = self._avg_price * self._position + price * qty
        new_qty = self._position + qty
        self._avg_price = total_cost / new_qty if new_qty > 0 else 0.0
        self._position = new_qty
        self._cash -= cost

    def sell(self, price: float, qty: float):
        if qty <= 0 or qty > self._position:
            return
        cost = self._position_cost(price, qty)
        proceeds = price * qty - cost
        self._position -= qty
        self._cash += proceeds
        if self._position == 0:
            self._avg_price = 0.0