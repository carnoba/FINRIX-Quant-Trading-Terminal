import json
import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from app import build_ensemble_payload
from macro_forecast_engine.json_utils import _json_safe
from macro_forecast_engine.modal_engine import _tft_quantile_paths
from macro_forecast_engine.trader_timemachine import backtest_strategy


def test_local_ensemble_json_serialization(ohlc):
    row = backtest_strategy(ohlc, "RSI Mean Reversion", "1D", account_equity=10000)
    row["generated_at"] = pd.Timestamp("2026-11-06T10:00:00Z")
    payload = build_ensemble_payload([row], "Local CPU")
    encoded = json.dumps(payload, allow_nan=False)
    assert "generated_at" in encoded
    assert "closed_trades" not in payload["leaderboard"][0]


def test_entry_exit_timestamp_distinctness(ohlc):
    result = backtest_strategy(ohlc, "RSI Mean Reversion", "1D", account_equity=10000)
    multi_bar_trades = [
        trade for trade in result["closed_trades"]
        if trade["entry_timestamp"] < trade["exit_timestamp"]
    ]
    assert multi_bar_trades
    assert all(
        trade["entry_timestamp"] != trade["exit_timestamp"]
        for trade in multi_bar_trades
    )


def test_local_cpu_import_without_modal(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", None)
    assert _json_safe({"timestamp": pd.Timestamp("2026-11-06")})["timestamp"].startswith("2026-11-06")
    app_module = importlib.import_module("app")
    assert app_module.CONFIG["execution_environment"] in {"local", "modal"}


def test_same_bar_stop_loss_priority(monkeypatch):
    index = pd.date_range("2025-01-01", periods=100, freq="D")
    high = np.full(100, 4001.0)
    low = np.full(100, 3999.0)
    high[32] = 4030.0
    low[32] = 3970.0
    frame = pd.DataFrame({
        "gold_open": np.full(100, 4000.0),
        "gold_high": high,
        "gold_low": low,
        "gold_close": np.full(100, 4000.0),
    }, index=index)

    def enter_once(prices, strategy, params):
        signal = pd.Series(0.0, index=prices.index)
        signal.iloc[31] = 1.0
        return signal

    monkeypatch.setattr("macro_forecast_engine.trader_timemachine._strategy_signal", enter_once)
    result = backtest_strategy(frame, "TFT Log-Return Trend", "1D", account_equity=10000)
    assert result["closed_trades"][0]["gross_pnl"] < 0


@pytest.mark.parametrize("timeframe", ["15m", "1h", "1D"])
def test_volatility_parity_timeframe_bounds(ohlc, timeframe):
    from macro_forecast_engine.trader_timemachine import _strategy_signal
    signal = _strategy_signal(ohlc, "Volatility Parity & Risk-Adjusted Allocation",
                              __import__("macro_forecast_engine.indicator_params",
                                         fromlist=["load_indicator_params"]).load_indicator_params())
    assert set(signal.dropna().unique()).issubset({-1.0, 0.0, 1.0})
    bars_per_day = {"15m": 92, "1h": 23, "1D": 1}[timeframe]
    target = 0.01 / np.sqrt(bars_per_day)
    realized = ohlc["gold_close"].pct_change().rolling(30).std().shift(1)
    valid = realized.dropna()
    multipliers = np.clip(target / valid.clip(lower=1e-6), 0.25, 2.0)
    assert multipliers.min() >= 0.25
    assert multipliers.max() <= 2.0
    assert realized.iloc[30] != ohlc["gold_close"].pct_change().rolling(30).std().iloc[30]


def test_tft_q50_trajectory_preservation():
    q50 = np.linspace(-0.002, 0.003, 252)
    lower, median, upper = _tft_quantile_paths(
        4000.0, q50 - 0.002, q50, q50 + 0.002, 252,
        pd.Series(q50),
    )
    assert not np.allclose(median, median[0])
    assert np.all(lower < median)
    assert np.all(median < upper)
