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


def analyze_news_sentiment(news: pd.DataFrame, model_name: str = "ProsusAI/finbert") -> dict[str, float]:
    """Score live news and expose a simple sentiment velocity metric."""
    from .data_loader import score_headlines

    if news.empty or "headline" not in news:
        return {"score": 0.0, "velocity": 0.0}
    score = score_headlines(news["headline"].tolist(), model_name=model_name)
    velocity = score - float(news["headline"].head(1).map(
        lambda value: score_headlines([value], model_name=model_name)
    ).iloc[0]) if len(news) > 1 else 0.0
    return {"score": float(score), "velocity": float(velocity)}
