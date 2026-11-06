"""Lookahead-safe, parameterized MT5/TradingView indicator replay utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .indicator_params import load_indicator_params

STRATEGIES = (
    "SMC Fair Value Gap (FVG)",
    "SMC Liquidity Order Block Breakout",
    "EMA Golden Cross / Death Cross",
    "Triple EMA Ribbon",
    "RSI Mean Reversion",
    "MACD Zero-Line & Signal Line Crossover",
    "Bollinger Bands Squeeze & Reversal",
    "Supertrend ATR Trend Follower",
    "Stochastic Oscillator",
    "ADX + DI Trend Classifier",
    "Ichimoku Cloud Kumo Breakout",
    "ATR Volatility Expansion Channel",
    "TFT Log-Return Trend",
    "Volatility Parity & Risk-Adjusted Allocation",
)

TIMEFRAME_PERIODS = {"1m": 390 * 252, "5m": 78 * 252, "15m": 26 * 252,
                     "1h": 6 * 252, "4h": 2 * 252, "1D": 252}
TIMEFRAME_FETCH = {
    "1m": ("7d", "1m"), "5m": ("60d", "5m"), "15m": ("60d", "15m"),
    "1h": ("60d", "1h"), "4h": ("60d", "1h"), "1D": ("730d", "1d"),
}


def normalize_timeframe_prices(prices: pd.Series | pd.DataFrame, timeframe: str) -> pd.Series | pd.DataFrame:
    if isinstance(prices, pd.DataFrame):
        frame = prices.dropna(how="all").sort_index()
        if timeframe == "4h":
            aggregation = {column: "last" for column in frame.columns}
            for name in ("gold_open", "open"):
                if name in frame:
                    aggregation[name] = "first"
            for name in ("gold_high", "high"):
                if name in frame:
                    aggregation[name] = "max"
            for name in ("gold_low", "low"):
                if name in frame:
                    aggregation[name] = "min"
            return frame.resample("4h").agg(aggregation).dropna(subset=[
                column for column in ("gold_close", "close") if column in frame
            ])
        return frame
    series = prices.dropna().astype(float).sort_index()
    return series.resample("4h").last().dropna() if timeframe == "4h" else series


PIP_SIZE = 0.10
CONTRACT_SIZE_OUNCES = 100.0
PIP_VALUE_PER_LOT = PIP_SIZE * CONTRACT_SIZE_OUNCES
SPREAD_PIPS = 1.5
SLIPPAGE_PIPS = 0.5


def _ohlc_data(prices: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(prices, pd.DataFrame):
        frame = prices.copy()
        columns = {str(column).lower(): column for column in frame.columns}
        close = frame[columns.get("close", columns.get("gold_close"))].astype(float)
        high = frame[columns.get("high", columns.get("gold_high", columns.get("close", columns.get("gold_close"))))].astype(float)
        low = frame[columns.get("low", columns.get("gold_low", columns.get("close", columns.get("gold_close"))))].astype(float)
        open_price = frame[columns.get("open", columns.get("gold_open", columns.get("close", columns.get("gold_close"))))].astype(float)
        return pd.DataFrame({"open": open_price, "high": high, "low": low, "close": close}, index=frame.index)
    close = pd.Series(prices, dtype=float)
    spread = close.diff().abs().rolling(5, min_periods=1).mean().fillna(close * 0.001).clip(lower=0.01)
    return pd.DataFrame({
        "open": close,
        "high": close + spread,
        "low": close - spread,
        "close": close,
    }, index=close.index)


def _rsi(prices: pd.Series, period: int) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _atr(prices: pd.Series | pd.DataFrame, period: int) -> pd.Series:
    frame = _ohlc_data(prices)
    previous_close = frame["close"].shift(1)
    true_range = pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - previous_close).abs(),
        (frame["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def _strategy_signal(prices: pd.Series | pd.DataFrame, strategy: str,
                     params: dict[str, dict[str, float | int]]) -> pd.Series:
    ohlc = _ohlc_data(prices)
    prices = ohlc["close"]
    returns = prices.pct_change()
    strategy = {
        "RSI 21 Overbought/Oversold": "RSI Mean Reversion",
        "EMA 50/200 Golden Cross": "EMA Golden Cross / Death Cross",
        "Supertrend Volatility Breakout": "Supertrend ATR Trend Follower",
        "Volatility Parity": "Volatility Parity & Risk-Adjusted Allocation",
    }.get(strategy, strategy)
    rsi = params["rsi"]
    ema = params["ema"]
    macd = params["macd"]
    bollinger = params["bollinger"]
    supertrend = params["supertrend"]
    stochastic = params["stochastic"]
    adx = params["adx"]
    atr = params["atr"]
    ichimoku = params["ichimoku"]
    smc = params["smc"]
    if strategy == "RSI Mean Reversion":
        value = _rsi(prices, int(rsi["period"]))
        return pd.Series(np.select([value < rsi["oversold"], value > rsi["overbought"]],
                                   [1.0, -1.0], default=0.0), index=prices.index)
    if strategy == "EMA Golden Cross / Death Cross":
        fast = prices.ewm(span=int(ema["fast_period"]), adjust=False).mean()
        slow = prices.ewm(span=int(ema["slow_period"]), adjust=False).mean()
        trend = prices.ewm(span=int(ema["trend_period"]), adjust=False).mean()
        return pd.Series(np.select([(fast > slow) & (fast > trend),
                                    (fast < slow) & (fast < trend)],
                                   [1.0, -1.0], default=0.0), index=prices.index)
    if strategy == "Triple EMA Ribbon":
        fast = prices.ewm(span=int(ema["fast_period"]), adjust=False).mean()
        medium = prices.ewm(span=int(ema["slow_period"]), adjust=False).mean()
        slow = prices.ewm(span=int(ema["trend_period"]), adjust=False).mean()
        return pd.Series(np.select([(fast > medium) & (medium > slow),
                                    (fast < medium) & (medium < slow)], [1.0, -1.0],
                                   default=0.0), index=prices.index)
    if strategy == "MACD Zero-Line & Signal Line Crossover":
        fast = prices.ewm(span=int(macd["fast"]), adjust=False).mean()
        slow = prices.ewm(span=int(macd["slow"]), adjust=False).mean()
        line = fast - slow
        signal = line.ewm(span=int(macd["signal"]), adjust=False).mean()
        return pd.Series(np.sign(line - signal) + np.sign(line), index=prices.index).clip(-1, 1)
    if strategy == "Bollinger Bands Squeeze & Reversal":
        mean = prices.rolling(int(bollinger["period"])).mean()
        deviation = prices.rolling(int(bollinger["period"])).std()
        return pd.Series(np.select([prices < mean - bollinger["std_dev"] * deviation,
                                    prices > mean + bollinger["std_dev"] * deviation],
                                   [1.0, -1.0], default=0.0), index=prices.index)
    if strategy == "Supertrend ATR Trend Follower":
        atr_value = _atr(ohlc, int(supertrend["atr_period"]))
        midpoint = (ohlc["high"] + ohlc["low"]) / 2.0
        upper_basic = midpoint + float(supertrend["factor"]) * atr_value
        lower_basic = midpoint - float(supertrend["factor"]) * atr_value
        upper = upper_basic.copy()
        lower = lower_basic.copy()
        trend = pd.Series(0.0, index=prices.index)
        for index in range(1, len(prices)):
            previous = prices.index[index - 1]
            current = prices.index[index]
            upper.loc[current] = (
                upper_basic.loc[current]
                if prices.loc[previous] > upper.loc[previous]
                else min(upper_basic.loc[current], upper.loc[previous])
            )
            lower.loc[current] = (
                lower_basic.loc[current]
                if prices.loc[previous] < lower.loc[previous]
                else max(lower_basic.loc[current], lower.loc[previous])
            )
            if trend.loc[previous] <= 0 and prices.loc[current] > upper.loc[previous]:
                trend.loc[current] = 1.0
            elif trend.loc[previous] >= 0 and prices.loc[current] < lower.loc[previous]:
                trend.loc[current] = -1.0
            else:
                trend.loc[current] = trend.loc[previous] or (
                    1.0 if prices.loc[current] >= midpoint.loc[current] else -1.0
                )
        return trend.fillna(0.0).clip(-1.0, 1.0)
    if strategy == "Stochastic Oscillator":
        k_period = int(stochastic["k_period"])
        low = prices.rolling(k_period).min()
        high = prices.rolling(k_period).max()
        k = 100 * (prices - low) / (high - low).replace(0, np.nan)
        d = k.rolling(int(stochastic["d_period"])).mean()
        return pd.Series(np.select([(k < 20) & (k > d), (k > 80) & (k < d)],
                                   [1.0, -1.0], default=0.0), index=prices.index)
    if strategy == "ADX + DI Trend Classifier":
        period = int(adx["period"])
        up_move = ohlc["high"].diff()
        down_move = -ohlc["low"].diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)
        previous_close = prices.shift(1)
        true_range = pd.concat([
            ohlc["high"] - ohlc["low"],
            (ohlc["high"] - previous_close).abs(),
            (ohlc["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr_value = true_range.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100.0 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_value
        minus_di = 100.0 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_value
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx_value = dx.ewm(alpha=1 / period, adjust=False).mean()
        return pd.Series(np.select([
            (adx_value >= adx["threshold"]) & (plus_di > minus_di),
            (adx_value >= adx["threshold"]) & (minus_di > plus_di),
        ], [1.0, -1.0], default=0.0), index=prices.index)
    if strategy == "Ichimoku Cloud Kumo Breakout":
        tenkan = int(ichimoku["tenkan"])
        kijun = int(ichimoku["kijun"])
        senkou_b = int(ichimoku["senkou_b"])
        conversion = (prices.rolling(tenkan).max() + prices.rolling(tenkan).min()) / 2
        base = (prices.rolling(kijun).max() + prices.rolling(kijun).min()) / 2
        span_a = ((conversion + base) / 2).shift(int(kijun))
        span_b = ((prices.rolling(senkou_b).max() + prices.rolling(senkou_b).min()) / 2).shift(int(kijun))
        cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)
        return pd.Series(np.select([prices > cloud_top, prices < cloud_bottom], [1.0, -1.0],
                                   default=0.0), index=prices.index)
    if strategy == "ATR Volatility Expansion Channel":
        period = int(atr["period"])
        mean = prices.ewm(span=period, adjust=False).mean()
        width = _atr(ohlc, period) * 1.5
        return pd.Series(np.select([prices > mean + width, prices < mean - width],
                                   [1.0, -1.0], default=0.0), index=prices.index)
    if strategy == "SMC Fair Value Gap (FVG)":
        gap = float(smc["fvg_min_pip_gap"]) * PIP_SIZE
        bullish = ohlc["high"].shift(2) < ohlc["low"] - gap
        bearish = ohlc["low"].shift(2) > ohlc["high"] + gap
        return pd.Series(np.select([bullish, bearish], [1.0, -1.0], default=0.0), index=prices.index)
    if strategy == "SMC Liquidity Order Block Breakout":
        lookback = int(smc["order_block_lookback"])
        upper = prices.rolling(lookback).max().shift(1)
        lower = prices.rolling(lookback).min().shift(1)
        return pd.Series(np.select([prices > upper, prices < lower], [1.0, -1.0],
                                   default=0.0), index=prices.index)
    if strategy == "Volatility Parity & Risk-Adjusted Allocation":
        volatility = returns.rolling(30).std().replace(0, np.nan)
        direction = np.sign(np.log1p(returns).rolling(30).mean())
        return direction.where(volatility.notna(), 0.0).fillna(0.0).clip(-1.0, 1.0)
    drift = np.log1p(returns).rolling(30).mean()
    return np.sign(drift).fillna(0.0)


def backtest_strategy(prices: pd.Series | pd.DataFrame, strategy: str, timeframe: str,
                      indicator_params: dict[str, Any] | None = None,
                      account_equity: float = 1000.0,
                      risk_per_trade_pct: float = 1.0) -> dict[str, float]:
    ohlc = _ohlc_data(prices).dropna().astype(float).sort_index()
    series = ohlc["close"]
    if len(series) < 80:
        raise ValueError(f"Not enough {timeframe} observations for {strategy}")
    params = indicator_params or load_indicator_params()
    signal = pd.Series(_strategy_signal(ohlc, strategy, params), index=series.index).shift(1).fillna(0.0)
    atr_period = int(params["atr"]["period"])
    stop_distance = _atr(ohlc, atr_period).shift(1)
    fallback_distance = series.shift(1).abs() * 0.001
    stop_distance = stop_distance.fillna(fallback_distance).clip(lower=0.01)
    pip_size = PIP_SIZE
    pip_value = PIP_VALUE_PER_LOT
    current_equity = max(float(account_equity), 0.01)
    starting_equity = current_equity
    equity_values = [current_equity]
    trade_pnls: list[float] = []
    trade_lots: list[float] = []
    transaction_costs = 0.0
    closed_trades: list[dict[str, float]] = []
    position: dict[str, float] | None = None
    for index in range(1, len(ohlc)):
        direction = float(signal.iloc[index])
        bar = ohlc.iloc[index]
        if position is None and direction:
            entry = float(bar["open"]) + (SLIPPAGE_PIPS * pip_size * direction)
            distance = max(float(stop_distance.iloc[index]), pip_size)
            risk_dollars = current_equity * max(float(risk_per_trade_pct), 0.0) / 100.0
            lot_size = risk_dollars / (distance * CONTRACT_SIZE_OUNCES)
            entry_cost = SPREAD_PIPS * pip_size * lot_size * CONTRACT_SIZE_OUNCES
            current_equity = max(current_equity - entry_cost, 0.01)
            transaction_costs += entry_cost
            position = {
                "direction": direction, "entry": entry, "stop": entry - direction * distance,
                "target": entry + direction * distance * 2.0, "lots": lot_size,
            }
        if position is not None:
            direction = position["direction"]
            stop_hit = (bar["low"] <= position["stop"] if direction > 0 else
                        bar["high"] >= position["stop"])
            target_hit = (bar["high"] >= position["target"] if direction > 0 else
                          bar["low"] <= position["target"])
            reversal = direction * float(signal.iloc[index]) < 0
            if stop_hit or target_hit or reversal or index == len(ohlc) - 1:
                exit_price = position["stop"] if stop_hit else position["target"] if target_hit else float(bar["close"])
                exit_price -= direction * (SPREAD_PIPS + SLIPPAGE_PIPS) * pip_size
                exit_cost = SPREAD_PIPS * pip_size * position["lots"] * CONTRACT_SIZE_OUNCES
                transaction_costs += exit_cost
                pnl = direction * (exit_price - position["entry"]) * position["lots"] * CONTRACT_SIZE_OUNCES
                pnl -= exit_cost
                current_equity = max(current_equity + pnl, 0.01)
                closed_trades.append({"pnl": pnl, "lots": position["lots"]})
                trade_pnls.append(pnl)
                trade_lots.append(position["lots"])
                position = None
        unrealized = 0.0
        if position is not None:
            unrealized = position["direction"] * (
                float(bar["close"]) - position["entry"]
            ) * position["lots"] * CONTRACT_SIZE_OUNCES
        equity_values.append(max(current_equity + unrealized, 0.01))
    equity = pd.Series(equity_values, index=range(len(equity_values)))
    trade_returns = pd.Series(trade_pnls, dtype=float)
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    drawdown = equity / equity.cummax() - 1.0
    strategy_returns = equity.pct_change().fillna(0.0)
    periods = {
        "15m": 23184,
        "1h": 5796,
        "4h": 1449,
        "1D": 252,
    }.get(timeframe, 252)
    volatility = float(strategy_returns.std())
    return {
        "strategy": strategy, "timeframe": timeframe, "observations": float(len(series)),
        "win_rate": float((trade_returns > 0).mean() * 100) if not trade_returns.empty else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if not losses.empty else float("inf"),
        "max_drawdown": float(abs(drawdown.min()) * 100),
        "sharpe_ratio": float(strategy_returns.mean() / volatility * np.sqrt(periods))
        if volatility > 0 else 0.0,
        "net_expected_return": float((current_equity - starting_equity) / starting_equity * 100.0),
        "account_equity": float(account_equity),
        "risk_per_trade_pct": float(risk_per_trade_pct),
        "final_equity": float(current_equity),
        "average_lot_size": float(np.mean(trade_lots)) if trade_lots else 0.0,
        "closed_trade_count": float(len(trade_pnls)),
        "transaction_costs": float(transaction_costs),
    }


def run_strategy_matrix(prices: pd.Series, strategies: list[str], timeframes: list[str],
                        indicator_params: dict[str, Any] | None = None,
                        account_equity: float = 1000.0,
                        risk_per_trade_pct: float = 1.0) -> pd.DataFrame:
    rows = [backtest_strategy(prices, strategy, timeframe, indicator_params,
                              account_equity, risk_per_trade_pct)
            for timeframe in timeframes for strategy in strategies]
    return pd.DataFrame(rows)


def strategy_confidence(row: dict[str, Any]) -> float:
    win_component = np.clip((float(row["win_rate"]) - 50.0) * 0.5, -20.0, 20.0)
    profit_component = np.clip((float(row["profit_factor"]) - 1.0) * 15.0, -15.0, 15.0)
    sharpe_component = np.clip(float(row["sharpe_ratio"]) * 8.0, -15.0, 20.0)
    drawdown_component = -np.clip(float(row["max_drawdown"]) * 0.1, 0.0, 15.0)
    return float(np.clip(55.0 + win_component + profit_component +
                         sharpe_component + drawdown_component, 0.0, 95.0))


def replay_forecasts(prices: pd.Series, forecast_fn: Callable[[pd.Series, int], pd.Series],
                     horizon: int, min_history: int = 60) -> pd.DataFrame:
    series = prices.dropna().sort_index().astype(float)
    rows: list[dict[str, Any]] = []
    for origin in range(max(min_history, 1), len(series) - horizon + 1):
        history = series.iloc[:origin].copy()
        forecast = pd.Series(forecast_fn(history, horizon)).reset_index(drop=True)
        actual = series.iloc[origin:origin + horizon].reset_index(drop=True)
        length = min(len(forecast), len(actual))
        if not length:
            continue
        errors = (forecast.iloc[:length] - actual.iloc[:length]).abs()
        rows.append({"origin": series.index[origin - 1], "horizon": length,
                     "actual": float(actual.iloc[length - 1]),
                     "forecast": float(forecast.iloc[length - 1]),
                     "mae": float(errors.mean()),
                     "direction_correct": bool((forecast.iloc[length - 1] - history.iloc[-1]) *
                                               (actual.iloc[length - 1] - history.iloc[-1]) >= 0)})
    return pd.DataFrame(rows)
