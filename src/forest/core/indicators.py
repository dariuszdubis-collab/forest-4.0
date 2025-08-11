from __future__ import annotations
import numpy as np
import pandas as pd

def ema(series: pd.Series, length: int) -> pd.Series:
    if length <= 0:
        raise ValueError("EMA length must be positive")
    return series.ewm(span=length, adjust=False).mean()

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    if length <= 0:
        raise ValueError("ATR length must be positive")
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    return tr.rolling(window=length, min_periods=1).mean()