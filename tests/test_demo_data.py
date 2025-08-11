from __future__ import annotations

from pathlib import Path

import pandas as pd

from forest4.examples.synthetic import generate_ohlc


def test_generate_ohlc_basic(tmp_path: Path):
    df = generate_ohlc(periods=200, freq="D", seed=1, start_price=50.0)

    # Struktura i podstawowe własności
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert set(["open", "high", "low", "close"]).issubset(df.columns)
    assert (df[["open", "high", "low", "close"]] > 0).all().all()

    # Relacje OHLC
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()

    # Zapis/odczyt CSV
    out = tmp_path / "demo.csv"
    df.to_csv(out)
    df2 = pd.read_csv(out, index_col=0, parse_dates=True)
    assert len(df2) == len(df)
    assert df2.index.equals(df.index)

