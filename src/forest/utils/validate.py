from __future__ import annotations

import pandas as pd

REQUIRED = ("open","high","low","close")

def ensure_backtest_ready(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("df is empty")
    # Normalize columns: try case-insensitive mapping to lower
    cols = {c.lower(): c for c in df.columns}
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")
    out = df.rename(columns={cols["open"]:"open", cols["high"]:"high",
                             cols["low"]:"low", cols["close"]:"close"}).copy()
    # Ensure datetime index
    if not isinstance(out.index, pd.DatetimeIndex):
        # try to parse index or common date column
        if "date" in out.columns:
            out.index = pd.to_datetime(out["date"], errors="coerce")
            out = out.drop(columns=["date"])
        else:
            out.index = pd.to_datetime(out.index, errors="coerce")
    if out.index.isna().any():
        raise ValueError("Found NaT in index after datetime conversion")
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out