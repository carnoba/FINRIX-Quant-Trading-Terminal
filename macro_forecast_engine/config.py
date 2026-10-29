"""Environment-backed configuration for the Modal forecast engine."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
FORECAST_MODE = os.getenv("FORECAST_MODE", os.getenv("FORECAST_HORIZON", "monthly")).lower()
FINBERT_MODEL = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
FRED_SERIES_TIPS = "DFII10"
FRED_SERIES_2Y = "DGS2"

HORIZONS = {"daily": 30, "monthly": 22, "yearly": 252}
if FORECAST_MODE not in HORIZONS:
    raise ValueError(f"FORECAST_MODE must be one of {', '.join(HORIZONS)}")
