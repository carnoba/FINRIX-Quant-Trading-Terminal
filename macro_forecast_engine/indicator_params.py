"""Persistent FINRIX MT5/TradingView indicator parameters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDICATOR_PARAMS_PATH = PROJECT_ROOT / "data" / "finrix_indicator_params.json"

DEFAULT_INDICATOR_PARAMS: dict[str, dict[str, float | int]] = {
    "rsi": {"period": 14, "overbought": 70, "oversold": 30},
    "ema": {"fast_period": 9, "slow_period": 21, "trend_period": 200},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "bollinger": {"period": 20, "std_dev": 2.0},
    "supertrend": {"atr_period": 10, "factor": 3.0},
    "stochastic": {"k_period": 14, "d_period": 3, "slowing": 3},
    "adx": {"period": 14, "threshold": 25},
    "atr": {"period": 14},
    "ichimoku": {"tenkan": 9, "kijun": 26, "senkou_b": 52},
    "smc": {"fvg_min_pip_gap": 1.5, "order_block_lookback": 20},
}

_LIMITS: dict[str, dict[str, tuple[float, float]]] = {
    "rsi": {"period": (2, 100), "overbought": (50, 99), "oversold": (1, 50)},
}


def _coerce(value: Any, default: float | int) -> float | int:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    if isinstance(default, int):
        return int(round(converted))
    return converted


def validate_indicator_params(params: dict[str, Any] | None) -> dict[str, dict[str, float | int]]:
    source = params or {}
    result: dict[str, dict[str, float | int]] = {}
    for group, defaults in DEFAULT_INDICATOR_PARAMS.items():
        result[group] = {}
        for key, default in defaults.items():
            value = _coerce(source.get(group, {}).get(key), default)
            low, high = _LIMITS.get(group, {}).get(key, (0.1, 1000.0))
            result[group][key] = min(high, max(low, value))
    result["rsi"]["oversold"] = min(result["rsi"]["oversold"], result["rsi"]["overbought"] - 1)
    result["rsi"]["overbought"] = max(result["rsi"]["overbought"], result["rsi"]["oversold"] + 1)
    return result


def load_indicator_params(path: Path = INDICATOR_PARAMS_PATH) -> dict[str, dict[str, float | int]]:
    try:
        params = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        params = {}
    validated = validate_indicator_params(params)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or validated != params:
        path.write_text(json.dumps(validated, indent=2), encoding="utf-8")
    return validated


def save_indicator_params(
    params: dict[str, Any],
    path: Path = INDICATOR_PARAMS_PATH,
) -> dict[str, dict[str, float | int]]:
    validated = validate_indicator_params(params)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validated, indent=2), encoding="utf-8")
    return validated
