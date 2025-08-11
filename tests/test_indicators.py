
import pandas as pd
from forest.core.indicators import ema, atr

def test_ema_simple():
    s = pd.Series([1,2,3,4,5])
    e = ema(s, 2)
    assert e.isna().sum() == 0
    assert e.iloc[-1] > e.iloc[0]

def test_atr_positive():
    import numpy as np
    high = pd.Series([10,11,12,13])
    low = pd.Series([9,9.5,10,11])
    close = pd.Series([9.5,10.5,11,12])
    a = atr(high, low, close, 3)
    assert (a >= 0).all()
