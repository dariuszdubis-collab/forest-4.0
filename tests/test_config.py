
import json
from pathlib import Path
from forest.config import BacktestSettings

def test_config_from_yaml(tmp_path: Path):
    cfg = {
        "symbol": "EURUSD",
        "timeframe": "H",
        "strategy": {"name":"ema_cross","fast":10,"slow":30},
        "risk": {"initial_capital": 50000.0, "risk_per_trade": 0.02}
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(
"""symbol: EURUSD
timeframe: H
strategy:
  name: ema_cross
  fast: 10
  slow: 30
risk:
  initial_capital: 50000.0
  risk_per_trade: 0.02
""", encoding="utf-8")
    s = BacktestSettings.from_file(p)
    assert s.symbol == "EURUSD"
    assert s.timeframe == "1h"
    assert s.strategy.fast == 10
    assert s.risk.initial_capital == 50000.0
