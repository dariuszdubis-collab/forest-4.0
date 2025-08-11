import pytest

from forest4.utils.timeframes import normalize_timeframe


def test_normalize_timeframe_variants():
    assert normalize_timeframe("H") == "1h"
    assert normalize_timeframe("1H") == "1h"
    assert normalize_timeframe("60min") == "1h"
    assert normalize_timeframe("1D") == "1d"


def test_invalid_timeframe():
    with pytest.raises(ValueError):
        normalize_timeframe("2Q")
