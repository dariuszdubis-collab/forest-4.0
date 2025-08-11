from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import BacktestSettings
from ..core.indicators import atr, ema
from ..utils.log import log
from ..utils.validate import ensure_backtest_ready
from .risk import RiskManager
from .tradebook import TradeBook


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list
    final_equity: float
    max_drawdown: float


def ema_cross_strategy(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    f = ema(df["close"], fast)
    s = ema(df["close"], slow)
    sig = np.where(f > s, 1, -1)
    sig[(f.isna()) | (s.isna())] = 0
    return pd.Series(sig, index=df.index, name="signal")


def run_backtest(df: pd.DataFrame, settings: BacktestSettings) -> BacktestResult:
    df = ensure_backtest_ready(df)
    # indicators
    df["signal"] = ema_cross_strategy(df, settings.strategy.fast, settings.strategy.slow)
    df["atr"] = atr(df["high"], df["low"], df["close"], length=settings.strategy.atr_length)

    risk = RiskManager(
        initial_capital=settings.risk.initial_capital,
        risk_per_trade=settings.risk.risk_per_trade,
        max_drawdown=settings.risk.max_drawdown,
        fee_perc=settings.risk.fee_perc,
        slippage_perc=settings.risk.slippage_perc,
    )
    tb = TradeBook()

    position = 0.0
    last_sig = 0
    for ts, row in df.iterrows():
        price = float(row["close"])
        sig = int(row["signal"])
        # mark-to-market equity each bar (ZMIANA: z ceny)
        risk.mark_price(price)

        if risk.exceeded_max_dd():
            log.warning("max_dd_exceeded", time=str(ts), equity=risk.equity)
            break

        # signal change
        if sig != last_sig:
            # close if we have a long and signal turns negative
            if position > 0 and sig <= 0:
                risk.sell(price=price, qty=position)
                tb.add(ts, price, position, "SELL")
                position = 0.0
                log.info("trade", action="SELL", time=str(ts), price=price)
            # open long on positive signal
            if sig > 0 and position == 0.0:
                qty = risk.position_size(
                    price=price,
                    atr=float(row["atr"]),
                    atr_multiple=settings.strategy.atr_multiple,
                )
                if qty > 0:
                    risk.buy(price=price, qty=qty)
                    tb.add(ts, price, qty, "BUY")
                    position = qty
                    log.info("trade", action="BUY", time=str(ts), price=price, qty=qty)
            last_sig = sig

    # finalize equity curve as Series
    eq = pd.Series(risk._equity_curve, name="equity")
    # compute max drawdown
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max
    max_dd = float(dd.min()) if len(dd) else 0.0
    return BacktestResult(
        equity_curve=eq,
        trades=tb.trades,
        final_equity=float(eq.iloc[-1]),
        max_drawdown=abs(max_dd),
    )
