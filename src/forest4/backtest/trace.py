from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DecisionTrace:
    time: Any
    symbol: str
    filters: dict[str, Any]
    final: str  # "BUY" | "SELL" | "WAIT"
