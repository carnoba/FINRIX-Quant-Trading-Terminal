import json

import pandas as pd

from app import build_ensemble_payload
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
