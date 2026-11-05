import json
import math

import numpy as np
import pytest

from macro_forecast_engine.data_loader import _cache_path
from macro_forecast_engine.indicator_params import (
    DEFAULT_INDICATOR_PARAMS,
    load_indicator_params,
    validate_indicator_params,
)
from macro_forecast_engine.modal_engine import _json_safe


def test_json_safe_coerces_numpy_scalars():
    value = _json_safe({"a": np.float64(1.5), "b": np.int64(3)})
    assert value == {"a": 1.5, "b": 3}
    assert isinstance(value["a"], float)


def test_json_safe_replaces_nonfinite_values():
    value = _json_safe({"nan": float("nan"), "inf": float("inf"), "neg": float("-inf")})
    assert value == {"nan": None, "inf": None, "neg": None}


def test_json_safe_handles_nested_lists():
    value = _json_safe([np.nan, {"x": np.inf}, [np.float64(2.0)]])
    assert value[0] is None
    assert value[1]["x"] is None
    assert value[2][0] == 2.0


def test_json_safe_is_json_serializable():
    value = _json_safe({"nan": float("nan"), "number": np.float64(4.0)})
    encoded = json.dumps(value, allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded


def test_corrupt_parameter_file_recovers(tmp_path):
    path = tmp_path / "params.json"
    path.write_text("{bad", encoding="utf-8")
    result = load_indicator_params(path)
    assert result["rsi"]["period"] == DEFAULT_INDICATOR_PARAMS["rsi"]["period"]
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["ema"]["trend_period"] == 200


def test_nonfinite_parameter_values_recover():
    result = validate_indicator_params({"rsi": {"period": "inf"}, "ema": {"slow_period": "NaN"}})
    assert result["rsi"]["period"] == 14
    assert result["ema"]["slow_period"] == 21


def test_cache_paths_are_timeframe_specific():
    assert _cache_path("15m").name == "gold_cache_15m.csv"
    assert _cache_path("1h").name == "gold_cache_1h.csv"
    assert _cache_path("1d").name == "gold_cache_1d.csv"


def test_default_params_are_finite():
    for group in DEFAULT_INDICATOR_PARAMS.values():
        for value in group.values():
            assert math.isfinite(float(value))


@pytest.mark.parametrize("payload", [
    {"rsi": {"period": -1}},
    {"rsi": {"period": 0}},
    {"rsi": {"period": 101}},
    {"ema": {"fast_period": "bad"}},
    {"macd": {"signal": None}},
    {"bollinger": {"std_dev": float("inf")}},
    {"supertrend": {"factor": float("nan")}},
    {"adx": {"threshold": -50}},
    {"atr": {"period": 0}},
    {"smc": {"order_block_lookback": 5000}},
])
def test_malicious_parameter_payloads_are_sanitized(payload):
    result = validate_indicator_params(payload)
    assert set(result) == set(DEFAULT_INDICATOR_PARAMS)
    assert all(math.isfinite(float(value)) for group in result.values() for value in group.values())
    assert result["rsi"]["oversold"] < result["rsi"]["overbought"]


@pytest.mark.parametrize("value", [
    float("nan"), float("inf"), float("-inf"), "NaN", "Infinity",
    "-Infinity", None, "", "1e309", "-1e309", "<script>",
    "1 OR 1=1", "; rm -rf", [], {}, -999999, 999999,
    0, 1, 14, 28, 100, 101, -1, 2.5,
])
def test_scalar_input_sanitization_matrix(value):
    result = validate_indicator_params({"rsi": {"period": value}})
    period = result["rsi"]["period"]
    assert 2 <= period <= 100
    assert math.isfinite(float(period))
    assert result["rsi"]["oversold"] < result["rsi"]["overbought"]
