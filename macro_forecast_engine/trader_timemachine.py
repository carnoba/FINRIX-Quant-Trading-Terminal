"""Lookahead-safe historical replay utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd


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
