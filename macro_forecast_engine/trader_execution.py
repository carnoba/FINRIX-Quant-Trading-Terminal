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
    account_equity: float = 10000.0,
) -> dict[str, Any]:
    """Combine FVG/liquidity anchors with quantile paths into trade levels.

    Key improvements:
    - Ideal Entry Zone anchored tightly around spot price (target_q50),
      not the full multi-day Q10-Q90 corridor.
    - Risk:Reward and Actual Risk % computed from entry_mid so daily
      setups produce realistic institutional ratios (typically 1:1.5 to 1:3.5).
    - Lot sizing and risk % reflect the 0.01 minimum lot floor for small accounts.
    """
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
    neutral = bias.lower() not in {"bullish", "bearish"}
    if neutral:
        entry_low = min(entry, float(lows[0]))
        entry_high = max(entry, float(highs[0]))
        # Tight entry zone anchored around spot price, not full Q10-Q90 corridor
        entry_spot = float(mids[0])
        corridor_half_width = max(
            abs(entry_spot - lows[0]), abs(highs[0] - entry_spot), 1e-9
        )
        entry_width = 0.15 * corridor_half_width
        entry_low = max(lows[0], entry_spot - entry_width)
        entry_high = min(highs[0], entry_spot + entry_width)
        entry_mid = (entry_low + entry_high) / 2.0
        entry_midpoint = (entry_low + entry_high) / 2.0
        targets = [entry_low - 25.0, entry_low - 50.0, entry_low - 75.0]
        stop = entry_high + buffer
        entry = entry_midpoint
    elif bullish:
        local_levels = [
            float(fvg.get("low", entry)),
            float(liquidity.get("ssl", entry)),
        ]
        structural_stop = min(
            (level for level in local_levels
             if entry * (1.0 - MAX_LOCAL_STRUCTURE_DISTANCE) <= level < entry),
            default=entry,
        )
        # Tight stop based on entry_mid structure, not full corridor
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
        # Tight stop based on entry_mid structure, not full corridor
        stop = max(entry + buffer, structural_stop + buffer if structural_stop > entry else entry + buffer)
        targets = [float(np.quantile(mids, quantile)) for quantile in (0.75, 0.5, 0.25)]
        targets = [min(entry, target) for target in targets]
    # --- Compute risk / reward from entry_mid ---
    sl_price = float(stop)
    tp1_price = float(targets[0])
    sl_distance = abs(sl_price - entry_mid)
    tp1_distance = abs(tp1_price - entry_mid)
    risk_reward = tp1_distance / max(sl_distance, 1e-9)

    # --- Lot sizing & Actual Risk % (reflects 0.01 minimum lot floor) ---
    lot_size = max(0.01, account_equity * 0.01 / (sl_distance * 100.0))

    return {
        "side": "long" if bullish else "short",
        "entry_zone": [round(entry_low, 4), round(entry_high, 4)],
        "stop_loss": round(stop, 4),
        "take_profit_1": round(targets[0], 4),
        "take_profit_2": round(targets[1], 4),
        "take_profit_3": round(targets[2], 4),
        "risk_distance": round(abs(entry - sl_price), 4),
        "reward_distance": round(tp1_distance, 4),  # from entry_mid
        "risk_reward": round(risk_reward, 4),  # tp1_distance / sl_distance from entry_mid
        "actual_risk_pct": round(lot_size * 100.0 * sl_distance / account_equity, 4),
        "lot_size": round(lot_size, 4),
        "trailing_stop": {
            "activation": round(targets[0], 4),
            "distance": round(abs(targets[0] - stop), 4),
        },
        "anchors": {"fvg": fvg, "liquidity": liquidity},
    }
