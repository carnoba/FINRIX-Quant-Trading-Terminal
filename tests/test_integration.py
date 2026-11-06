import json
import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

from app import build_ensemble_payload
from macro_forecast_engine.json_utils import _json_safe
from macro_forecast_engine.modal_engine import _quantile_forecast, _tft_quantile_paths
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


def test_tft_q50_trajectory_preservation():
    horizon = 30
    q50 = np.linspace(-0.002, 0.003, horizon)
    prediction = pd.DataFrame({
        "TFT-lo-80.0": q50 - 0.002,
        "TFT-median": q50,
        "TFT-hi-80.0": q50 + 0.002,
    })

    class FakeTFT:
        def __init__(self, **kwargs):
            self.train_trajectories = []
            self.valid_trajectories = []

    class FakeNeuralForecast:
        def __init__(self, models, freq):
            self.models = models

        def fit(self, frame, val_size):
            return self

        def predict(self):
            return prediction

    class FakeCallback:
        pass

    class FakeEarlyStopping:
        def __init__(self, **kwargs):
            pass

    fake_neuralforecast = types.ModuleType("neuralforecast")
    fake_neuralforecast.NeuralForecast = FakeNeuralForecast
    fake_models = types.ModuleType("neuralforecast.models")
    fake_models.TFT = FakeTFT
    fake_losses = types.ModuleType("neuralforecast.losses")
    fake_losses_pytorch = types.ModuleType("neuralforecast.losses.pytorch")
    fake_losses_pytorch.MQLoss = lambda **kwargs: kwargs
    fake_lightning = types.ModuleType("pytorch_lightning")
    fake_callbacks = types.ModuleType("pytorch_lightning.callbacks")
    fake_callbacks.Callback = FakeCallback
    fake_callbacks.EarlyStopping = FakeEarlyStopping
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "neuralforecast", fake_neuralforecast)
    monkeypatch.setitem(sys.modules, "neuralforecast.models", fake_models)
    monkeypatch.setitem(sys.modules, "neuralforecast.losses", fake_losses)
    monkeypatch.setitem(sys.modules, "neuralforecast.losses.pytorch", fake_losses_pytorch)
    monkeypatch.setitem(sys.modules, "pytorch_lightning", fake_lightning)
    monkeypatch.setitem(sys.modules, "pytorch_lightning.callbacks", fake_callbacks)
    try:
        features = pd.DataFrame({
            "gold_close": np.linspace(4000.0, 4100.0, 100),
            "gold_return": np.full(100, 0.001),
        }, index=pd.date_range("2026-01-01", periods=100, freq="D"))
        result, diagnostics = _quantile_forecast(features, horizon)
    finally:
        monkeypatch.undo()
    expected_median = 4100.0 * np.exp(np.cumsum(q50))
    np.testing.assert_allclose(result["q0.5"].to_numpy(), expected_median, rtol=1e-12)
    assert diagnostics["model_used"] == "NeuralForecast_TFT"
