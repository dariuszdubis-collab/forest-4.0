from __future__ import annotations

import re

_ALIASES = {
    "H": "1h",
    "1H": "1h",
    "D": "1d",
    "1D": "1d",
    "W": "1w",
    "1W": "1w",
    "M": "1mo",
    "1M": "1mo",
    "60min": "1h",
    "30min": "30m",
    "15min": "15m",
    "5min": "5m",
    "1min": "1m",
}

_VALID = {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1mo"}


def normalize_timeframe(tf: str) -> str:
    if not isinstance(tf, str) or not tf:
        raise ValueError("timeframe must be a non-empty string")
    tf = tf.strip()
    if tf in _ALIASES:
        return _ALIASES[tf]
    # patterns like '1h','4H','15m','1D'
    m = re.fullmatch(r"(\d+)([mMhHdDwW])", tf)
    if m:
        num, unit = m.groups()
        unit = unit.lower()
        if unit == "m":
            norm = f"{num}m"
        elif unit == "h":
            norm = f"{num}h"
        elif unit == "d":
            norm = f"{num}d"
        elif unit == "w":
            norm = f"{num}w"
        else:
            raise ValueError(f"Unsupported unit in timeframe: {tf}")
        # collapse common minutes/hours/days to a known set
        if norm in _VALID:
            return norm
    if tf.lower() in _VALID:
        return tf.lower()
    raise ValueError(f"Unsupported timeframe: {tf}")
