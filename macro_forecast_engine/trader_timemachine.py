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
    "1h": ("730d", "1h"), "4h": ("730d", "1h"), "1D": ("10y", "1d"),
}


def normalize_timeframe_prices(prices: pd.Series, timeframe: str) -> pd.Series:
    series = prices.dropna().astype(float).sort_index()
    return series.resample("4h").last().dropna() if timeframe == "4h" else series


def _rsi(prices: pd.Series, period: int) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _atr(prices: pd.Series, period: int) -> pd.Series:
    return prices.diff().abs().ewm(alpha=1 / period, adjust=False).mean()


def _strategy_signal(prices: pd.Series, strategy: str,
                     params: dict[str, dict[str, float | int]]) -> pd.Series:
    prices = prices.astype(float)
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
        return np.sign(fast - slow).where((fast > trend) | (fast < trend), 0.0)
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
        mean = prices.rolling(int(supertrend["atr_period"])).mean()
        width = _atr(prices, int(supertrend["atr_period"])) * supertrend["factor"]
        return np.sign(prices - mean - width).where(prices.notna(), 0.0)
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
        movement = prices.diff()
        strength = movement.abs().ewm(alpha=1 / period, adjust=False).mean()
        adx_value = strength / prices.ewm(alpha=1 / period, adjust=False).mean() * 100
        return np.sign(movement).where(adx_value >= adx["threshold"], 0.0)
    if strategy == "Ichimoku Cloud Kumo Breakout":
        tenkan = int(ichimoku["tenkan"])
        kijun = int(ichimoku["kijun"])
        senkou_b = int(ichimoku["senkou_b"])
        conversion = (prices.rolling(tenkan).max() + prices.rolling(tenkan).min()) / 2
        base = (prices.rolling(kijun).max() + prices.rolling(kijun).min()) / 2
        cloud = ((prices.rolling(senkou_b).max() + prices.rolling(senkou_b).min()) / 2 + base) / 2
        return pd.Series(np.select([prices > cloud, prices < cloud], [1.0, -1.0],
                                   default=0.0), index=prices.index)
    if strategy == "ATR Volatility Expansion Channel":
        period = int(atr["period"])
        mean = prices.ewm(span=period, adjust=False).mean()
        width = _atr(prices, period) * 1.5
        return pd.Series(np.select([prices > mean + width, prices < mean - width],
                                   [1.0, -1.0], default=0.0), index=prices.index)
    if strategy == "SMC Fair Value Gap (FVG)":
        gap = float(smc["fvg_min_pip_gap"]) / 10000
        bullish = prices > prices.shift(2) * (1 + gap)
        bearish = prices < prices.shift(2) * (1 - gap)
        return pd.Series(np.select([bullish, bearish], [1.0, -1.0], default=0.0), index=prices.index)
    if strategy == "SMC Liquidity Order Block Breakout":
        lookback = int(smc["order_block_lookback"])
        upper = prices.rolling(lookback).max().shift(1)
        lower = prices.rolling(lookback).min().shift(1)
        return pd.Series(np.select([prices > upper, prices < lower], [1.0, -1.0],
                                   default=0.0), index=prices.index)
    if strategy == "Volatility Parity & Risk-Adjusted Allocation":
        volatility = returns.rolling(30).std().replace(0, np.nan)
        return (np.sign(np.log1p(returns).rolling(30).mean()) /
                (volatility * np.sqrt(252))).clip(-2.0, 2.0).fillna(0.0)
    drift = np.log1p(returns).rolling(30).mean()
    return np.sign(drift).fillna(0.0)


def backtest_strategy(prices: pd.Series, strategy: str, timeframe: str,
                      indicator_params: dict[str, Any] | None = None,
                      account_equity: float = 1000.0,
                      risk_per_trade_pct: float = 1.0) -> dict[str, float]:
    series = prices.dropna().astype(float).sort_index()
    if len(series) < 80:
        raise ValueError(f"Not enough {timeframe} observations for {strategy}")
    params = indicator_params or load_indicator_params()
    signal = pd.Series(_strategy_signal(series, strategy, params), index=series.index).shift(1).fillna(0.0)
    returns = series.pct_change().fillna(0.0)
    strategy_returns = (signal * returns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    equity = (1.0 + strategy_returns).cumprod()
    wins = strategy_returns[strategy_returns > 0]
    losses = strategy_returns[strategy_returns < 0]
    drawdown = equity / equity.cummax() - 1.0
    periods = TIMEFRAME_PERIODS.get(timeframe, 252)
    volatility = float(strategy_returns.std())
    return {
        "strategy": strategy, "timeframe": timeframe, "observations": float(len(series)),
        "win_rate": float((strategy_returns > 0).mean() * 100),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if not losses.empty else float("inf"),
        "max_drawdown": float(abs(drawdown.min()) * 100),
        "sharpe_ratio": float(strategy_returns.mean() / volatility * np.sqrt(periods))
        if volatility > 0 else 0.0,
        "net_expected_return": float((equity.iloc[-1] - 1.0) * 100),
        "account_equity": float(account_equity),
        "risk_per_trade_pct": float(risk_per_trade_pct),
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
