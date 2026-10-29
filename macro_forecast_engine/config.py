"""Environment-backed configuration for the Modal forecast engine."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
FORECAST_HORIZON = os.getenv("FORECAST_HORIZON", "monthly")
FINBERT_MODEL = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
FRED_SERIES_TIPS = os.getenv("FRED_SERIES_TIPS", "DFII10")

if FORECAST_HORIZON not in {"monthly", "yearly"}:
    raise ValueError("FORECAST_HORIZON must be 'monthly' or 'yearly'")
