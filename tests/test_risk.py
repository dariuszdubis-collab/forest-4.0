from forest4.backtest.risk import RiskManager


def test_position_size():
    rm = RiskManager(initial_capital=100000.0, risk_per_trade=0.01)
    qty = rm.position_size(price=100.0, atr=2.0, atr_multiple=2.0)
    assert qty > 0
    assert abs(qty - 250) < 1e-6


def test_max_dd_guard():
    rm = RiskManager(initial_capital=1000.0, risk_per_trade=0.1)
    for p in [1000, 800, 700, 600]:
        rm.record_mark_to_market(p)
    assert rm.exceeded_max_dd()
