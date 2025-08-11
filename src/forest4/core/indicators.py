from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    if length <= 0:
        raise ValueError("EMA length must be positive")
    return series.ewm(span=length, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    if length <= 0:
        raise ValueError("ATR length must be positive")
    prev_close = close.shift(1)

    # Składniki True Range; max po kolumnach z pominięciem NaN (skipna=True)
    tr_components = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1, skipna=True)

    # Średnia krocząca ATR (od 1 punktu, żeby uniknąć NaN w teście)
    return tr.rolling(window=length, min_periods=1).mean()
