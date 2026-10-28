"""Macro bias integration for the existing M15 RSI strategy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_forecast(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_macro_bias(signal: pd.Series, forecast: dict[str, Any]) -> pd.Series:
    """Keep RSI entries only when they agree with the macro month-end bias."""
    bias = str(forecast.get("month_end", {}).get("directional_bias", "neutral")).lower()
    filtered = signal.copy().astype(int)
    if bias == "bullish":
        filtered.loc[filtered < 0] = 0
    elif bias == "bearish":
        filtered.loc[filtered > 0] = 0
    return filtered.rename("macro_filtered_signal")


def filter_gold_rsi_entries(
    data: pd.DataFrame,
    forecast_path: Path | str,
    rsi_length: int = 21,
    oversold: float = 30,
    overbought: float = 70,
) -> pd.Series:
    """Generate the existing RSI signal and apply forecast directional gating."""
    close = data["close"].astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / rsi_length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_length, adjust=False).mean()
    rsi = 100.0 - 100.0 / (1.0 + gain / loss.replace(0.0, pd.NA))
    signal = pd.Series(0, index=data.index, dtype=int)
    signal.loc[rsi < oversold] = 1
    signal.loc[rsi > overbought] = -1
    return apply_macro_bias(signal, load_forecast(forecast_path))

