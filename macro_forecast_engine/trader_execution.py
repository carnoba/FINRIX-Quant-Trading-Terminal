"""SMC and TFT forecast-path execution levels."""

from __future__ import annotations

from typing import Any

import numpy as np

PIP_SIZE = 1.0
STOP_BUFFER_PIPS = 15
MAX_LOCAL_STRUCTURE_DISTANCE = 0.02


def build_execution_plan(
    entry: float,
    forecast: list[dict[str, Any]],
    smc: dict[str, Any],
    bias: str,
) -> dict[str, Any]:
    """Combine FVG/liquidity anchors with quantile paths into trade levels."""
    if not forecast:
        raise ValueError("forecast must contain at least one path point")
    entry = float(entry)
    bullish = bias.lower() == "bullish"
    lows = np.array([float(row["expected_low_q10"]) for row in forecast])
    mids = np.array([float(row["target_q50"]) for row in forecast])
    highs = np.array([float(row["expected_high_q90"]) for row in forecast])
    fvg = smc.get("fair_value_gaps", [{}])[0]
    liquidity = smc.get("liquidity_pools", {})
    buffer = PIP_SIZE * STOP_BUFFER_PIPS
    if bullish:
        local_levels = [
            float(fvg.get("low", entry)),
            float(liquidity.get("ssl", entry)),
        ]
        structural_stop = min(
            (level for level in local_levels
             if entry * (1.0 - MAX_LOCAL_STRUCTURE_DISTANCE) <= level < entry),
            default=entry,
        )
        stop = min(entry - buffer, structural_stop - buffer if structural_stop < entry else entry - buffer)
        targets = [float(np.quantile(mids, quantile)) for quantile in (0.75, 0.5, 0.25)]
        targets = [max(entry, target) for target in targets]
    else:
        local_levels = [
            float(fvg.get("high", entry)),
            float(liquidity.get("bsl", entry)),
        ]
        structural_stop = max(
            (level for level in local_levels
             if entry < level <= entry * (1.0 + MAX_LOCAL_STRUCTURE_DISTANCE)),
            default=entry,
        )
        stop = max(entry + buffer, structural_stop + buffer if structural_stop > entry else entry + buffer)
        targets = [float(np.quantile(mids, quantile)) for quantile in (0.75, 0.5, 0.25)]
        targets = [min(entry, target) for target in targets]
    risk_distance = abs(entry - stop)
    reward_distance = abs(targets[0] - entry)
    return {
        "side": "long" if bullish else "short",
        "entry_zone": [round(min(entry, float(lows[0])), 4), round(max(entry, float(highs[0])), 4)],
        "stop_loss": round(stop, 4),
        "take_profit_1": round(targets[0], 4),
        "take_profit_2": round(targets[1], 4),
        "take_profit_3": round(targets[2], 4),
        "risk_distance": round(risk_distance, 4),
        "reward_distance": round(reward_distance, 4),
        "risk_reward": round(reward_distance / max(risk_distance, 1e-9), 4),
        "trailing_stop": {
            "activation": round(targets[0], 4),
            "distance": round(abs(targets[0] - stop), 4),
        },
        "anchors": {"fvg": fvg, "liquidity": liquidity},
    }
