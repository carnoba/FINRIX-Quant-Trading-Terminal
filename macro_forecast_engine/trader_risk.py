"""Risk and position sizing helpers for forecast-driven execution."""

from __future__ import annotations

from typing import Any

import numpy as np

GOLD_PIP_SIZE = 0.10
GOLD_PIP_VALUE_PER_STANDARD_LOT = 1.00
MIN_STANDARD_LOT = 0.01


def calculate_lot_size(
    equity: float,
    risk_per_trade_pct: float,
    stop_loss_pips: float,
    pip_value_per_lot: float = GOLD_PIP_VALUE_PER_STANDARD_LOT,
    minimum_lot: float = MIN_STANDARD_LOT,
) -> float:
    """Calculate standard-lot size from current equity and dollar risk."""
    equity = max(float(equity), 0.0)
    risk_dollars = equity * max(float(risk_per_trade_pct), 0.0) / 100.0
    stop_loss_pips = max(float(stop_loss_pips), 1e-9)
    pip_value_per_lot = max(float(pip_value_per_lot), 1e-9)
    lots = risk_dollars / (stop_loss_pips * pip_value_per_lot)
    return max(float(minimum_lot), lots) if equity > 0 else 0.0

def size_position(
    entry: float,
    q10: float,
    q50: float,
    q90: float,
    equity: float,
    risk_fraction: float = 0.01,
    kelly_fraction: float = 0.25,
    micro_lot_floor: float = 0.01,
) -> dict[str, Any]:
    """Return fractional-Kelly and volatility-parity units from forecast corridors."""
    entry = float(entry)
    lower, median, upper = map(float, (q10, q50, q90))
    equity = max(float(equity), 0.0)
    downside = max(entry - lower, entry * 1e-4, 1e-9)
    upside = max(upper - entry, 0.0, 1e-9)
    win_probability = float(np.clip((median - lower) / max(upper - lower, 1e-9), 0.05, 0.95))
    reward_risk = upside / downside
    kelly = max(0.0, win_probability - (1.0 - win_probability) / max(reward_risk, 1e-9))
    fractional_kelly = min(1.0, kelly * float(kelly_fraction))
    corridor_volatility = max((upper - lower) / max(2.0 * entry, 1e-9), 1e-6)
    volatility_parity = min(1.0, float(risk_fraction) / corridor_volatility)
    allocation_fraction = min(fractional_kelly, volatility_parity)
    allocation = equity * allocation_fraction
    units = allocation / entry if entry > 0 else 0.0
    if equity > 0 and entry > 0:
        units = max(float(micro_lot_floor), units)
        allocation = max(allocation, units * entry)
    stop_loss_pips = downside / GOLD_PIP_SIZE
    lot_size = calculate_lot_size(
        equity, float(risk_fraction) * 100.0, stop_loss_pips,
        minimum_lot=micro_lot_floor,
    )
    return {
        "entry": round(entry, 4),
        "win_probability": round(win_probability, 6),
        "reward_risk": round(max(reward_risk, 0.0), 6),
        "fractional_kelly": round(fractional_kelly, 6),
        "corridor_volatility": round(corridor_volatility, 6),
        "volatility_parity": round(volatility_parity, 6),
        "allocation_fraction": round(allocation_fraction, 6),
        "allocation": round(allocation, 4),
        "units": round(units, 6),
        "lot_size": round(lot_size, 6),
        "pip_size": GOLD_PIP_SIZE,
        "pip_value_per_lot": GOLD_PIP_VALUE_PER_STANDARD_LOT,
        "stop_loss_pips": round(stop_loss_pips, 4),
    }
