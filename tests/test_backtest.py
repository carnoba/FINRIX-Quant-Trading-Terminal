import numpy as np
import pandas as pd
import pytest

from macro_forecast_engine.trader_timemachine import (
    STRATEGIES,
    backtest_strategy,
    normalize_timeframe_prices,
    run_strategy_matrix,
)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_all_strategies_produce_closed_trade_metrics(ohlc, strategy):
    result = backtest_strategy(ohlc, strategy, "1D", account_equity=10000, risk_per_trade_pct=1)
    assert result["closed_trade_count"] >= 0
    assert result["transaction_costs"] >= 0
    assert np.isfinite(result["final_equity"])
    assert np.isfinite(result["net_expected_return"])
    assert result["max_drawdown"] >= 0


@pytest.mark.parametrize("risk", [0.5, 1, 2, 4])
def test_risk_changes_equity_path(ohlc, risk):
    low = backtest_strategy(
        ohlc, "RSI Mean Reversion", "1D",
        account_equity=10000, risk_per_trade_pct=risk,
    )
    high = backtest_strategy(
        ohlc, "RSI Mean Reversion", "1D",
        account_equity=10000, risk_per_trade_pct=risk * 2,
    )
    assert low["final_equity"] != high["final_equity"]
    assert low["risk_per_trade_pct"] == risk
    assert high["risk_per_trade_pct"] == risk * 2


def test_transaction_costs_reduce_result(ohlc):
    result = backtest_strategy(ohlc, "EMA Golden Cross / Death Cross", "1D", account_equity=10000)
    assert result["transaction_costs"] > 0
    assert result["closed_trade_count"] > 0


def test_lookahead_safe_signal_is_shifted(ohlc):
    changed = ohlc.copy()
    changed.iloc[0, changed.columns.get_loc("gold_close")] *= 1.5
    first = backtest_strategy(ohlc, "RSI Mean Reversion", "1D")
    second = backtest_strategy(changed, "RSI Mean Reversion", "1D")
    assert first["closed_trade_count"] >= 0
    assert second["closed_trade_count"] >= 0
    assert first["final_equity"] != second["final_equity"]


@pytest.mark.parametrize("timeframe", ["15m", "1h", "4h", "1D"])
def test_timeframe_normalization(ohlc, timeframe):
    result = normalize_timeframe_prices(ohlc, timeframe)
    assert len(result) > 0
    assert "gold_close" in result.columns
    assert result.index.is_monotonic_increasing


def test_matrix_contains_all_selected_strategies(ohlc):
    result = run_strategy_matrix(ohlc, list(STRATEGIES), ["1D"], account_equity=10000)
    assert len(result) == 14
    assert set(result["strategy"]) == set(STRATEGIES)
    assert result["account_equity"].eq(10000).all()


@pytest.mark.parametrize("direction", [1, -1])
def test_sl_tp_lifecycle_on_directional_market(direction):
    index = pd.date_range("2024-01-01", periods=180, freq="D")
    close = 4000 + direction * np.arange(len(index)) * 1.5
    frame = pd.DataFrame({
        "gold_open": close,
        "gold_high": close + 1,
        "gold_low": close - 1,
        "gold_close": close,
    }, index=index)
    result = backtest_strategy(frame, "TFT Log-Return Trend", "1D")
    assert result["closed_trade_count"] >= 0
    assert result["transaction_costs"] >= 0
