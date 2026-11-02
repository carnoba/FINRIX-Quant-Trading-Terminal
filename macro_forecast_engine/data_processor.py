"""Macro sentiment and institutional flow feature aggregation."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def add_sentiment_and_flow(
    frame: pd.DataFrame,
    sentiment: float,
    cot: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Add central-bank sentiment and CFTC-derived institutional flow features."""
    result = frame.copy()
    result["central_bank_sentiment"] = float(sentiment)
    result["institutional_flow"] = 0.0
    if cot is not None and not cot.empty:
        columns = [name for name in ("commercial_positions", "noncommercial_positions")
                   if name in result]
        if columns:
            result["institutional_flow"] = result[columns].diff().sum(axis=1).fillna(0.0)
    return result
