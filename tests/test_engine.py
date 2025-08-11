import pandas as pd

from forest4.backtest.engine import run_backtest
from forest4.config import BacktestSettings


def test_run_backtest_small():
    data = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        },
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
    )
    settings = BacktestSettings()
    res = run_backtest(data, settings)
    assert res.final_equity > 0
    assert 0.0 <= res.max_drawdown <= 1.0
