import numpy as np
import pandas as pd
import pytest

from macro_forecast_engine.indicator_params import load_indicator_params, validate_indicator_params
from macro_forecast_engine.trader_timemachine import STRATEGIES, _strategy_signal


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_indicator_signal_shape_and_domain(ohlc, strategy):
    params = load_indicator_params()
    signal = _strategy_signal(ohlc, strategy, params)
    assert len(signal) == len(ohlc)
    assert signal.index.equals(ohlc.index)
    assert np.isfinite(signal.fillna(0)).all()


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_indicator_has_warmup_safe_values(ohlc, strategy):
    params = load_indicator_params()
    signal = _strategy_signal(ohlc, strategy, params)
    assert signal.notna().sum() > 0
    assert signal.abs().max() <= 2


@pytest.mark.parametrize("period", [2, 7, 14, 28, 50, 100])
def test_rsi_period_override_changes_validated_configuration(period):
    params = validate_indicator_params({"rsi": {"period": period}})
    assert params["rsi"]["period"] == period
    assert 2 <= params["rsi"]["period"] <= 100


@pytest.mark.parametrize("group,key,value", [
    ("ema", "fast_period", 5), ("ema", "slow_period", 34),
    ("macd", "fast", 8), ("macd", "signal", 5),
    ("bollinger", "std_dev", 3.0), ("supertrend", "factor", 2.0),
    ("adx", "threshold", 30), ("atr", "period", 21),
])
def test_indicator_override_values_are_persisted_in_memory(group, key, value):
    params = validate_indicator_params({group: {key: value}})
    assert params[group][key] == value
    assert np.isfinite(params[group][key])


def test_fvg_requires_three_bar_structure():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    frame = pd.DataFrame({
        "gold_open": [100, 100, 100, 110, 110],
        "gold_high": [101, 101, 101, 111, 111],
        "gold_low": [99, 99, 99, 109, 109],
        "gold_close": [100, 100, 100, 110, 110],
    }, index=index)
    signal = _strategy_signal(frame, "SMC Fair Value Gap (FVG)", load_indicator_params())
    assert signal.iloc[0] == 0
    assert signal.iloc[1] == 0
    assert signal.iloc[2] == 0


def test_ema_requires_trend_alignment(ohlc):
    signal = _strategy_signal(ohlc, "EMA Golden Cross / Death Cross", load_indicator_params())
    assert signal.isna().sum() == 0
    assert signal.abs().max() <= 1
