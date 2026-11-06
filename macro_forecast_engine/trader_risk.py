"""Risk and position sizing helpers for forecast-driven execution."""

from __future__ import annotations

from typing import Any

import numpy as np

GOLD_PIP_SIZE = 0.10
GOLD_CONTRACT_SIZE_OUNCES = 100.0
GOLD_PIP_VALUE_PER_STANDARD_LOT = GOLD_PIP_SIZE * GOLD_CONTRACT_SIZE_OUNCES
MIN_STANDARD_LOT = 0.01


def calculate_lot_size(
    equity: float,
    risk_per_trade_pct: float,
    sl_distance_dollars: float,
    minimum_lot: float = MIN_STANDARD_LOT,
) -> float:
    """Calculate standard lots from equity, risk percentage, and price distance."""
    equity = max(float(equity), 0.0)
    risk_dollars = equity * max(float(risk_per_trade_pct), 0.0) / 100.0
    sl_distance_dollars = max(float(sl_distance_dollars), 1e-9)
    lots = risk_dollars / (sl_distance_dollars * GOLD_CONTRACT_SIZE_OUNCES)
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
    stop_loss: float | None = None,
) -> dict[str, Any]:
    """Return fractional-Kelly and volatility-parity sizing in standard lots."""
    entry = float(entry)
    lower, median, upper = map(float, (q10, q50, q90))
    equity = max(float(equity), 0.0)
    downside = max(
        abs(entry - float(stop_loss)) if stop_loss is not None else entry - lower,
        entry * 1e-4,
        1e-9,
    )
    upside = max(upper - entry, 0.0, 1e-9)
    win_probability = float(np.clip((median - lower) / max(upper - lower, 1e-9), 0.05, 0.95))
    reward_risk = upside / downside
    kelly = max(0.0, win_probability - (1.0 - win_probability) / max(reward_risk, 1e-9))
    fractional_kelly = min(1.0, kelly * float(kelly_fraction))
    corridor_volatility = max((upper - lower) / max(2.0 * entry, 1e-9), 1e-6)
    volatility_parity = min(1.0, float(risk_fraction) / corridor_volatility)
    allocation_fraction = min(fractional_kelly, volatility_parity)
    allocation = equity * allocation_fraction
    lot_size = calculate_lot_size(
        equity,
        float(risk_fraction) * 100.0,
        downside,
        minimum_lot=micro_lot_floor,
    )
    ounces = lot_size * GOLD_CONTRACT_SIZE_OUNCES
    units = ounces
    if equity > 0 and entry > 0:
        allocation = max(allocation, ounces * entry)
    stop_loss_pips = downside / GOLD_PIP_SIZE
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
        "ounces": round(ounces, 6),
        "lot_size": round(lot_size, 6),
        "pip_size": GOLD_PIP_SIZE,
        "pip_value_per_lot": GOLD_PIP_VALUE_PER_STANDARD_LOT,
        "stop_loss_pips": round(stop_loss_pips, 4),
    }
