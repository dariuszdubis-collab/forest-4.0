from __future__ import annotations
from dataclasses import dataclass
from typing import List, Any

@dataclass
class Trade:
    time: Any
    price: float
    qty: float
    side: str  # "BUY" or "SELL"

class TradeBook:
    def __init__(self) -> None:
        self.trades: List[Trade] = []

    def add(self, time, price: float, qty: float, side: str):
        self.trades.append(Trade(time=time, price=price, qty=qty, side=side))