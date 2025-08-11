from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from .utils.timeframes import normalize_timeframe


class StrategySettings(BaseModel):
    name: Literal["ema_cross"] = "ema_cross"
    fast: int = 12
    slow: int = 26
    atr_length: int = 14
    atr_multiple: float = 2.0


class RiskSettings(BaseModel):
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.01  # 1%
    max_drawdown: float = 0.3  # 30%
    fee_perc: float = 0.0005  # 0.05% prowizji
    slippage_perc: float = 0.0


class BacktestSettings(BaseModel):
    symbol: str = "TEST"
    timeframe: str = "1h"
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)

    @field_validator("timeframe")
    @classmethod
    def _norm_tf(cls, v: str) -> str:
        return normalize_timeframe(v)

    @classmethod
    def from_file(cls, path: str | Path) -> "BacktestSettings":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        if p.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        elif p.suffix.lower() == ".json":
            import json

            data = json.loads(p.read_text(encoding="utf-8"))
        else:
            raise ValueError(f"Unsupported config extension: {p.suffix}")
        return cls(**data)