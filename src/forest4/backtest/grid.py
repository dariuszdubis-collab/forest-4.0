from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from ..config import BacktestSettings, StrategySettings
from .engine import run_backtest


@dataclass
class GridResult:
    fast: int
    slow: int
    final_equity: float
    max_dd: float
    cagr: float
    sharpe: float


def _compute_metrics(equity: pd.Series) -> tuple[float, float, float]:
    if equity.empty:
        return 0.0, 0.0, 0.0
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    final_ret = (end / start) - 1.0 if start else 0.0
    # approximate annualization by trading days
    duration_days = max(1, len(equity))
    years = max(1e-9, duration_days / 252)  # ~252 trading days
    cagr = (end / start) ** (1 / years) - 1 if start else 0.0
    rets = equity.pct_change().dropna()
    sharpe = np.sqrt(252) * (rets.mean() / (rets.std() + 1e-12)) if not rets.empty else 0.0
    return final_ret, cagr, sharpe


def param_grid(fast_values: Iterable[int], slow_values: Iterable[int]) -> list[tuple[int, int]]:
    return [(f, s) for f, s in product(fast_values, slow_values) if f < s]


def run_grid(
    df: pd.DataFrame,
    base: BacktestSettings,
    fast_values: Iterable[int],
    slow_values: Iterable[int],
) -> pd.DataFrame:
    rows = []
    for fast, slow in param_grid(fast_values, slow_values):
        settings = base.model_copy()
        settings.strategy = StrategySettings(
            name="ema_cross",
            fast=fast,
            slow=slow,
            atr_length=base.strategy.atr_length,
            atr_multiple=base.strategy.atr_multiple,
        )
        res = run_backtest(df, settings)
        final_ret, cagr, sharpe = _compute_metrics(res.equity_curve)
        rows.append(
            {
                "fast": fast,
                "slow": slow,
                "final_equity": res.final_equity,
                "max_dd": res.max_drawdown,
                "final_return": final_ret,
                "cagr": cagr,
                "sharpe": sharpe,
            }
        )
    return pd.DataFrame(rows)
