"""Central configuration for the Macro-Forecast Engine.

Loads settings from environment variables via python-dotenv.
Set variables in .env file or export them before running.
"""

import os
from pathlib import Path

# Base directory resolved relative to this file
BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# API Keys — loaded from environment variables (never hardcode!)
# ------------------------------------------------------------
# FRED_API_KEY: Required for FRED API access (economic data, TIPS yields)
FRED_API_KEY = os.getenv("FRED_API_KEY")

# MODAL_TOKEN: Required for Modal GPU deployment
MODAL_TOKEN = os.getenv("MODAL_TOKEN")

# ALPHA_VANTAGE_KEY: Optional, for additional market data
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")

# ------------------------------------------------------------
# Data Directories
# ------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = BASE_DIR / "cache"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Model Configuration
# ------------------------------------------------------------
# Forecast horizon: "monthly" or "yearly"
FORECAST_HORIZON = os.getenv("FORECAST_HORIZON", "monthly")

# NeuralForecast TFT (Temporal Fusion Transformer) parameters
TFT_HIDDEN_SIZE = int(os.getenv("TFT_HIDDEN_SIZE", "64"))
TFT_LAGS = int(os.getenv("TFT_LAGS", "24"))
TFT_OUTPUT_SIZE = int(os.getenv("TFT_OUTPUT_SIZE", "1"))

# FinBERT sentiment model name
FINBERT_MODEL = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")

# FRED TIPS Yields series identifier
FRED_SERIES_TIPS = os.getenv("FRED_SERIES_TIPS", "TIPSYIELD")

# ------------------------------------------------------------
# Execution Configuration
# ------------------------------------------------------------
# Default timeframe for signal generation
DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "M15")

# Risk management parameters
MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "2.0"))
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "0.1"))

# ------------------------------------------------------------
# CLI Mode Configuration
# ------------------------------------------------------------
# Supported modes: "monthly", "yearly"
MODE = os.getenv("MODE", "monthly")

# Validate mode
assert MODE in ("monthly", "yearly"), f"Invalid MODE: {MODE}. Use 'monthly' or 'yearly'."