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
) -> dict[str, Any]:
    """Calculate standard lots and report when the broker floor exceeds target risk."""
    equity = max(float(equity), 0.0)
    sl_distance_dollars = max(float(sl_distance_dollars), 1e-9)
    requested_risk_pct = float(risk_per_trade_pct)
    if requested_risk_pct <= 0.0 or equity <= 0.0:
        return {"lot_size": 0.0, "actual_risk_pct": 0.0, "warning": None}
    risk_dollars = equity * requested_risk_pct / 100.0
    raw_lot_size = risk_dollars / (sl_distance_dollars * GOLD_CONTRACT_SIZE_OUNCES)
    lot_size = max(float(minimum_lot), raw_lot_size)
    actual_risk_pct = (
        lot_size * sl_distance_dollars * GOLD_CONTRACT_SIZE_OUNCES / equity * 100.0
    )
    warning = None
    if actual_risk_pct > requested_risk_pct * 1.05:
        warning = (
            "Minimum lot floor exceeds requested risk; account equity is too small "
            "for this stop-loss distance."
        )
    return {
        "lot_size": float(lot_size),
        "actual_risk_pct": float(actual_risk_pct),
        "warning": warning,
    }

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
    lot_result = calculate_lot_size(
        equity,
        float(risk_fraction) * 100.0,
        downside,
        minimum_lot=micro_lot_floor,
    )
    lot_size = float(lot_result["lot_size"])
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
        "actual_risk_pct": round(float(lot_result["actual_risk_pct"]), 6),
        "risk_warning": lot_result["warning"],
    }
