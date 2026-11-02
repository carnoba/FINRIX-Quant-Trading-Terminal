"""Risk and position sizing helpers for forecast-driven execution."""

from __future__ import annotations

from typing import Any

import numpy as np


def size_position(
    entry: float,
    q10: float,
    q50: float,
    q90: float,
    equity: float,
    risk_fraction: float = 0.01,
    kelly_fraction: float = 0.25,
) -> dict[str, Any]:
    """Return fractional-Kelly and volatility-parity units from forecast corridors."""
    entry = float(entry)
    lower, median, upper = map(float, (q10, q50, q90))
    equity = max(float(equity), 0.0)
    downside = max(entry - lower, entry * 1e-4)
    upside = max(upper - entry, 0.0)
    win_probability = float(np.clip((median - lower) / max(upper - lower, 1e-9), 0.05, 0.95))
    reward_risk = upside / downside
    kelly = max(0.0, win_probability - (1.0 - win_probability) / max(reward_risk, 1e-9))
    fractional_kelly = min(1.0, kelly * float(kelly_fraction))
    corridor_volatility = max((upper - lower) / max(2.0 * entry, 1e-9), 1e-6)
    volatility_parity = min(1.0, float(risk_fraction) / corridor_volatility)
    allocation_fraction = min(fractional_kelly, volatility_parity)
    allocation = equity * allocation_fraction
    units = allocation / entry if entry > 0 else 0.0
    return {
        "entry": round(entry, 4),
        "win_probability": round(win_probability, 6),
        "reward_risk": round(reward_risk, 6),
        "fractional_kelly": round(fractional_kelly, 6),
        "corridor_volatility": round(corridor_volatility, 6),
        "volatility_parity": round(volatility_parity, 6),
        "allocation_fraction": round(allocation_fraction, 6),
        "allocation": round(allocation, 4),
        "units": round(units, 6),
    }
