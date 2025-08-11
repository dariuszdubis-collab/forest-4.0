from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class OrderRouter(Protocol):
    def connect(self) -> None: ...
    def close(self) -> None: ...
    def set_price(self, price: float) -> None: ...
    def market_order(self, side: str, qty: float, price: float | None = None) -> OrderResult: ...
    def position_qty(self) -> float: ...
    def equity(self) -> float: ...


@dataclass
class OrderResult:
    id: int
    status: str
    filled_qty: float
    avg_price: float
    error: str | None = None


class PaperBroker(OrderRouter):
    def __init__(self, fee_perc: float = 0.0005):
        self._fee = fee_perc
        self._cash = 100_000.0
        self._pos = 0.0
        self._last_price: float | None = None
        self._connected = False
        self._id = 0

    def connect(self) -> None:
        self._connected = True

    def close(self) -> None:
        self._connected = False

    def set_price(self, price: float) -> None:
        self._last_price = price

    def position_qty(self) -> float:
        return self._pos

    def equity(self) -> float:
        price = self._last_price or 0.0
        return self._cash + self._pos * price

    def market_order(self, side: str, qty: float, price: float | None = None) -> OrderResult:
        self._id += 1
        if not self._connected:
            return OrderResult(self._id, "rejected", 0.0, 0.0, "not connected")
        px = price if price is not None else self._last_price
        if px is None:
            return OrderResult(self._id, "rejected", 0.0, 0.0, "no price")
        if qty <= 0:
            return OrderResult(self._id, "rejected", 0.0, 0.0, "invalid qty")

        gross = qty * px
        fee = gross * self._fee

        if side.upper() == "BUY":
            if gross + fee > self._cash:
                return OrderResult(self._id, "rejected", 0.0, 0.0, "insufficient cash")
            self._cash -= gross + fee
            self._pos += qty
            return OrderResult(self._id, "filled", qty, px)
        if side.upper() == "SELL":
            if qty > self._pos:
                return OrderResult(self._id, "rejected", 0.0, 0.0, "insufficient position")
            self._pos -= qty
            self._cash += gross - fee
            return OrderResult(self._id, "filled", qty, px)
        return OrderResult(self._id, "rejected", 0.0, 0.0, "unknown side")
