from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DemoSpec:
    start: str = "2024-01-01"
    periods: int = 500
    freq: str = "D"
    seed: int = 42
    start_price: float = 100.0


def generate_ohlc(
    start: str = "2024-01-01",
    periods: int = 500,
    freq: str = "D",
    seed: int = 42,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """
    Generuje syntetyczne dane OHLC:
    - geometryczny spacer losowy z lekkim dodatnim dryfem,
    - open = close poprzedniej świecy,
    - high/low wyznaczane wokół open/close.

    Zwraca DataFrame z kolumnami: open, high, low, close i indeks DatetimeIndex.
    """
    if periods <= 1:
        raise ValueError("`periods` must be > 1")
    if start_price <= 0:
        raise ValueError("`start_price` must be positive")

    rng = np.random.default_rng(seed)
    # Dzienny dryf ~0.05% i zmienność ~1%
    drift = 0.0005
    vol = 0.01

    rets = drift + vol * rng.standard_normal(periods)
    prices = np.empty(periods, dtype=float)
    prices[0] = start_price
    for i in range(1, periods):
        prices[i] = prices[i - 1] * (1.0 + rets[i])

    idx = pd.date_range(start=start, periods=periods, freq=freq)

    close = pd.Series(prices, index=idx, name="close")
    open_ = close.shift(1).fillna(close.iloc[0])
    # amplituda knotów zależna od zmienności chwili
    wiggle = np.abs(vol * 2.5 * rng.standard_normal(periods))
    high = pd.Series(np.maximum(open_.values, close.values) * (1.0 + wiggle), index=idx, name="high")
    low = pd.Series(np.minimum(open_.values, close.values) * (1.0 - wiggle), index=idx, name="low")

    df = pd.DataFrame(
        {
            "open": open_.astype(float),
            "high": high.astype(float),
            "low": low.astype(float),
            "close": close.astype(float),
        },
        index=idx,
    )

    # sanity: low <= min(open,close) <= max(open,close) <= high
    bad = (df["low"] > df[["open", "close"]].min(axis=1)) | (df["high"] < df[["open", "close"]].max(axis=1))
    if bad.any():
        raise RuntimeError("Generated inconsistent OHLC values")

    return df


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate synthetic OHLC CSV for Forest 4.0 demos.")
    p.add_argument("--start", default="2024-01-01", help="Datetime start (e.g., 2024-01-01)")
    p.add_argument("--periods", type=int, default=500, help="Number of rows/candles")
    p.add_argument("--freq", default="D", help="Pandas offset alias (e.g., D, H, 15min)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--start-price", type=float, default=100.0, help="Starting price")
    p.add_argument("--out", default="", help="Path to output CSV (if empty, prints head())")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    df = generate_ohlc(
        start=args.start,
        periods=args.periods,
        freq=args.freq,
        seed=args.seed,
        start_price=args.start_price,
    )
    if args.out:
        df.to_csv(args.out, index=True, date_format="%Y-%m-%d %H:%M:%S")
        print(f"Wrote {len(df):,} rows to {args.out}")
    else:
        # szybki podgląd
        print(df.head(10).to_string())


if __name__ == "__main__":
    main()

