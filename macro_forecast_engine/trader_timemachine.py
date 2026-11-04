"""Lookahead-safe historical replay utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import numpy as np

import pandas as pd

STRATEGIES = (
    "SMC Fair Value Gap (FVG)",
    "TFT Log-Return Trend",
    "RSI 21 Overbought/Oversold",
    "EMA 50/200 Golden Cross",
    "Supertrend Volatility Breakout",
    "Volatility Parity",
)
TIMEFRAME_PERIODS = {"1m": 390 * 252, "5m": 78 * 252, "15m": 26 * 252,
                     "1h": 6 * 252, "4h": 2 * 252, "1D": 252}
TIMEFRAME_FETCH = {
    "1m": ("7d", "1m"), "5m": ("60d", "5m"), "15m": ("60d", "15m"),
    "1h": ("730d", "1h"), "4h": ("730d", "1h"), "1D": ("10y", "1d"),
}


def normalize_timeframe_prices(prices: pd.Series, timeframe: str) -> pd.Series:
    """Convert fetched bars to the requested replay timeframe."""
    series = prices.dropna().astype(float).sort_index()
    if timeframe == "4h":
        return series.resample("4h").last().dropna()
    return series


def _strategy_signal(prices: pd.Series, strategy: str) -> pd.Series:
    prices = prices.astype(float)
    returns = prices.pct_change()
    if strategy == "RSI 21 Overbought/Oversold":
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(21).mean()
        loss = (-delta.clip(upper=0)).rolling(21).mean()
        rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        return pd.Series(np.select([rsi < 30, rsi > 70], [1.0, -1.0], default=0.0), index=prices.index)
    if strategy == "EMA 50/200 Golden Cross":
        return pd.Series(np.sign(prices.ewm(span=50, adjust=False).mean()
                                 - prices.ewm(span=200, adjust=False).mean()), index=prices.index)
    if strategy == "Supertrend Volatility Breakout":
        mean = returns.rolling(20).mean()
        deviation = returns.rolling(20).std().replace(0, np.nan)
        return pd.Series(np.select([returns > mean + deviation, returns < mean - deviation],
                                   [1.0, -1.0], default=0.0), index=prices.index)
    if strategy == "SMC Fair Value Gap (FVG)":
        bullish = prices > prices.shift(2) * 1.002
        bearish = prices < prices.shift(2) * 0.998
        return pd.Series(np.select([bullish, bearish], [1.0, -1.0], default=0.0), index=prices.index)
    drift = np.log1p(returns).rolling(30).mean()
    signal = np.sign(drift).fillna(0.0)
    if strategy == "Volatility Parity":
        volatility = returns.rolling(30).std().replace(0, np.nan)
        return (signal / (volatility * np.sqrt(252))).clip(-2.0, 2.0).fillna(0.0)
    return signal


def backtest_strategy(prices: pd.Series, strategy: str, timeframe: str) -> dict[str, float]:
    """Run a no-lookahead strategy replay and calculate institutional metrics."""
    series = prices.dropna().astype(float).sort_index()
    if len(series) < 80:
        raise ValueError(f"Not enough {timeframe} observations for {strategy}")
    signal = _strategy_signal(series, strategy).shift(1).fillna(0.0)
    returns = series.pct_change().fillna(0.0)
    strategy_returns = (signal * returns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    equity = (1.0 + strategy_returns).cumprod()
    wins = strategy_returns[strategy_returns > 0]
    losses = strategy_returns[strategy_returns < 0]
    drawdown = equity / equity.cummax() - 1.0
    periods = TIMEFRAME_PERIODS.get(timeframe, 252)
    volatility = float(strategy_returns.std())
    return {
        "strategy": strategy,
        "timeframe": timeframe,
        "observations": float(len(series)),
        "win_rate": float((strategy_returns > 0).mean() * 100),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if not losses.empty else float("inf"),
        "max_drawdown": float(abs(drawdown.min()) * 100),
        "sharpe_ratio": float(strategy_returns.mean() / volatility * np.sqrt(periods))
        if volatility > 0 else 0.0,
        "net_expected_return": float((equity.iloc[-1] - 1.0) * 100),
    }


def run_strategy_matrix(prices: pd.Series, strategies: list[str], timeframes: list[str]) -> pd.DataFrame:
    rows = [backtest_strategy(prices, strategy, timeframe)
            for timeframe in timeframes for strategy in strategies]
    return pd.DataFrame(rows)


def strategy_confidence(row: dict[str, Any]) -> float:
    """Bound confidence to a stable 0-100 score from risk-adjusted metrics."""
    win_component = np.clip((float(row["win_rate"]) - 50.0) * 0.5, -20.0, 20.0)
    profit_component = np.clip((float(row["profit_factor"]) - 1.0) * 15.0, -15.0, 15.0)
    sharpe_component = np.clip(float(row["sharpe_ratio"]) * 8.0, -15.0, 20.0)
    drawdown_component = -np.clip(float(row["max_drawdown"]) * 0.1, 0.0, 15.0)
    return float(np.clip(55.0 + win_component + profit_component
                         + sharpe_component + drawdown_component, 0.0, 95.0))


def replay_forecasts(
    prices: pd.Series,
    forecast_fn: Callable[[pd.Series, int], pd.Series],
    horizon: int,
    min_history: int = 60,
) -> pd.DataFrame:
    """Replay forecasts using only observations available at each origin."""
    series = prices.dropna().sort_index().astype(float)
    rows: list[dict[str, Any]] = []
    for origin in range(max(min_history, 1), len(series) - horizon + 1):
        history = series.iloc[:origin].copy()
        forecast = pd.Series(forecast_fn(history, horizon)).reset_index(drop=True)
        actual = series.iloc[origin:origin + horizon].reset_index(drop=True)
        if forecast.empty:
            continue
        length = min(len(forecast), len(actual))
        errors = (forecast.iloc[:length] - actual.iloc[:length]).abs()
        rows.append({
            "origin": series.index[origin - 1],
            "horizon": length,
            "actual": float(actual.iloc[length - 1]),
            "forecast": float(forecast.iloc[length - 1]),
            "mae": float(errors.mean()),
            "direction_correct": bool(
                (forecast.iloc[length - 1] - history.iloc[-1])
                * (actual.iloc[length - 1] - history.iloc[-1]) >= 0
            ),
        })
    return pd.DataFrame(rows)
