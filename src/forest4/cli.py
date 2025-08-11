from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .backtest.engine import run_backtest
from .backtest.grid import run_grid
from .config import BacktestSettings


def _cmd_backtest(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.input)
    settings = BacktestSettings()
    settings.strategy.fast = args.fast
    settings.strategy.slow = args.slow
    settings.risk.initial_capital = args.capital

    res = run_backtest(df, settings)

    # krótki raport na stdout
    print(
        f"[forest4] backtest ok | final_equity={res.final_equity:.2f} "
        f"| max_dd={res.max_drawdown:.4f}"
    )

    # zapis equity do CSV (opcjonalnie)
    if args.out:
        eq_df = res.equity_curve.reset_index()
        eq_df.columns = ["step", "equity"]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        eq_df.to_csv(args.out, index=False)
        print(f"[forest4] equity saved -> {args.out}")
    return 0


def _cmd_grid(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.input)
    base = BacktestSettings()

    fast_vals = range(args.fast_min, args.fast_max + 1, args.fast_step)
    slow_vals = range(args.slow_min, args.slow_max + 1, args.slow_step)

    out = run_grid(df, base, fast_vals, slow_vals)
    print(f"[forest4] grid rows={len(out)}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"[forest4] grid saved -> {args.out}")
    else:
        # pokaż pierwsze kilka wierszy
        print(out.head())

    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forest4",
        description="FOREST 4.0 CLI – backtest & grid",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # backtest
    pb = sub.add_parser("backtest", help="uruchom pojedynczy backtest (EMA cross)")
    pb.add_argument("--input", required=True, type=str, help="plik CSV z OHLC")
    pb.add_argument("--fast", type=int, default=12)
    pb.add_argument("--slow", type=int, default=26)
    pb.add_argument("--capital", type=float, default=100_000.0)
    pb.add_argument("--out", type=str, default=None, help="zapisz equity do CSV")
    pb.set_defaults(func=_cmd_backtest)

    # grid
    pg = sub.add_parser("grid", help="uruchom grid search (EMA fast/slow)")
    pg.add_argument("--input", required=True, type=str, help="plik CSV z OHLC")
    pg.add_argument("--fast-min", type=int, default=5)
    pg.add_argument("--fast-max", type=int, default=15)
    pg.add_argument("--fast-step", type=int, default=1)
    pg.add_argument("--slow-min", type=int, default=20)
    pg.add_argument("--slow-max", type=int, default=60)
    pg.add_argument("--slow-step", type=int, default=1)
    pg.add_argument("--out", type=str, default=None, help="zapisz wyniki do CSV")
    pg.set_defaults(func=_cmd_grid)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


# wygodne entrypointy skryptowe
def main_backtest() -> int:
    return main(["backtest", *sys.argv[1:]])


def main_grid() -> int:
    return main(["grid", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
