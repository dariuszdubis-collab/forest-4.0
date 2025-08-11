import pandas as pd

from forest4.cli import main


def _make_csv(tmp_path):
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        {
            "open": list(range(100, 130)),
            "high": [x + 1 for x in range(100, 130)],
            "low": [x - 1 for x in range(100, 130)],
            "close": list(range(100, 130)),
        },
        index=idx,
    )
    p = tmp_path / "data.csv"
    df.reset_index().rename(columns={"index": "time"}).to_csv(p, index=False)
    return p


def test_cli_backtest_runs(tmp_path):
    src = _make_csv(tmp_path)
    out = tmp_path / "eq.csv"
    code = main(["backtest", "--input", str(src), "--fast", "3", "--slow", "5", "--out", str(out)])
    assert code == 0
    assert out.exists()
    df = pd.read_csv(out)
    assert "equity" in df.columns
    assert len(df) > 0


def test_cli_grid_runs(tmp_path):
    src = _make_csv(tmp_path)
    out = tmp_path / "grid.csv"
    code = main(
        [
            "grid",
            "--input",
            str(src),
            "--fast-min",
            "3",
            "--fast-max",
            "4",
            "--slow-min",
            "5",
            "--slow-max",
            "6",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    df = pd.read_csv(out)
    assert {"fast", "slow", "final_equity"}.issubset(df.columns)
    assert len(df) >= 1
