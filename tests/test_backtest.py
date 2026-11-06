import numpy as np
import pandas as pd
import pytest

from macro_forecast_engine.trader_timemachine import (
    CONTRACT_SIZE_OUNCES,
    PIP_SIZE,
    PIP_VALUE_PER_LOT,
    STRATEGIES,
    SPREAD_PIPS,
    SLIPPAGE_PIPS,
    _strategy_signal,
    backtest_strategy,
    normalize_timeframe_prices,
    run_strategy_matrix,
)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_all_strategies_produce_closed_trade_metrics(ohlc, strategy):
    result = backtest_strategy(ohlc, strategy, "1D", account_equity=10000, risk_per_trade_pct=1)
    assert result["closed_trade_count"] == int(result["closed_trade_count"])
    assert result["final_equity"] > 0
    assert result["net_expected_return"] == pytest.approx(
        (result["final_equity"] / result["account_equity"] - 1.0) * 100.0
    )
    assert (result["transaction_costs"] > 0) == (result["closed_trade_count"] > 0)
    assert np.isfinite(result["max_drawdown"])


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
    assert first["final_equity"] != second["final_equity"]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_true_lookahead_safety_prefix_is_identical(ohlc, strategy):
    params = __import__(
        "macro_forecast_engine.indicator_params",
        fromlist=["load_indicator_params"],
    ).load_indicator_params()
    full = _strategy_signal(ohlc, strategy, params).fillna(0.0)
    truncated = _strategy_signal(ohlc.iloc[:-50], strategy, params).fillna(0.0)
    pd.testing.assert_series_equal(
        truncated,
        full.iloc[:-50],
        check_names=False,
        check_exact=True,
    )


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
    assert result["closed_trade_count"] == int(result["closed_trade_count"])
    assert result["final_equity"] > 0
    assert (result["transaction_costs"] > 0) == (result["closed_trade_count"] > 0)


def test_backtest_execution_truncation(ohlc):
    full = backtest_strategy(ohlc, "RSI Mean Reversion", "1D", account_equity=10000)
    truncated = backtest_strategy(ohlc.iloc[:-50], "RSI Mean Reversion", "1D", account_equity=10000)
    cutoff = ohlc.index[-50]
    full_history = [
        trade for trade in full["closed_trades"]
        if trade["exit_timestamp"] < cutoff
    ]
    truncated_history = [
        trade for trade in truncated["closed_trades"]
        if trade["exit_timestamp"] < cutoff
    ]
    assert len(full_history) == len(truncated_history)
    for expected, actual in zip(full_history, truncated_history):
        assert expected["entry_timestamp"] == actual["entry_timestamp"]
        assert expected["exit_timestamp"] == actual["exit_timestamp"]
        assert expected["pnl"] == pytest.approx(actual["pnl"], abs=1e-10)


def test_exact_single_spread_deduction(monkeypatch):
    index = pd.date_range("2025-01-01", periods=100, freq="D")
    frame = pd.DataFrame({
        "gold_open": 4000.0, "gold_high": 4000.0,
        "gold_low": 4000.0, "gold_close": 4000.0,
    }, index=index)

    def one_flat_trade(prices, strategy, params):
        signal = pd.Series(0.0, index=prices.index)
        signal.iloc[0] = 1.0
        return signal

    monkeypatch.setattr(
        "macro_forecast_engine.trader_timemachine._strategy_signal",
        one_flat_trade,
    )
    result = backtest_strategy(frame, "TFT Log-Return Trend", "1D", account_equity=1000)
    trade = result["closed_trades"][0]
    assert trade["gross_pnl"] == pytest.approx(0.0)
    assert trade["transaction_cost"] / trade["lots"] / PIP_VALUE_PER_LOT == pytest.approx(
        SPREAD_PIPS + SLIPPAGE_PIPS
    )
    assert trade["pnl"] == pytest.approx(-trade["transaction_cost"])


@pytest.mark.parametrize("timeframe", ["15m", "1h", "1D"])
def test_volatility_parity_timeframe_bounds(ohlc, timeframe):
    result = backtest_strategy(
        ohlc, "Volatility Parity & Risk-Adjusted Allocation", timeframe,
        account_equity=10000, risk_per_trade_pct=1,
    )
    assert 0.25 <= result["volatility_multiplier_min"] <= 2.0
    assert 0.25 <= result["volatility_multiplier_max"] <= 2.0


def test_volatility_parity_entry_bar_immunity(monkeypatch, ohlc):
    entry_index = 31
    variants = []
    for move in (0.0, 4.0, 40.0):
        variant = ohlc.copy()
        variant.iloc[entry_index, variant.columns.get_loc("gold_close")] += move
        variants.append(variant)

    def enter_once(prices, strategy, params):
        signal = pd.Series(0.0, index=prices.index)
        signal.iloc[entry_index - 1] = 1.0
        return signal

    monkeypatch.setattr(
        "macro_forecast_engine.trader_timemachine._strategy_signal", enter_once,
    )
    results = [
        backtest_strategy(
            variant, "Volatility Parity & Risk-Adjusted Allocation", "1D",
            account_equity=10000, risk_per_trade_pct=1,
        )
        for variant in variants
    ]
    lots = [result["closed_trades"][0]["lots"] for result in results]
    assert lots[1] / lots[0] == pytest.approx(1.0)
    assert lots[2] / lots[0] == pytest.approx(1.0)
