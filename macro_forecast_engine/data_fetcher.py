"""Compatibility exports for timeframe-aware market data ingestion."""

from .data_loader import (
    GOLD_TICKERS,
    INTRADAY_PERIOD_LIMITS,
    fetch_market_data,
)

__all__ = ["GOLD_TICKERS", "INTRADAY_PERIOD_LIMITS", "fetch_market_data"]
