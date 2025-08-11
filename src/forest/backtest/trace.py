from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DecisionTrace:
    time: Any
    symbol: str
    filters: Dict[str, Any]
    final: str  # "BUY" | "SELL" | "WAIT"