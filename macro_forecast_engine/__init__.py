"""Configuration for Macro-Forecast Engine - environment variables loaded via dotenv"""
from pathlib import Path

# Resolve paths relative to this file's location
BASE_DIR = Path(__file__).resolve().parent

# --- API Keys (loaded from environment variables via python-dotenv) ---
# These are NOT hardcoded - they are loaded from .env file at runtime
# FRED_API_KEY = os.getenv("FRED_API_KEY")  # Loaded by consuming scripts
# MODAL_TOKEN = os.getenv("MODAL_TOKEN")  # Loaded by consuming scripts
# ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")  # Loaded by consuming scripts

# --- Data Paths ---
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = BASE_DIR / "cache"

# --- Model Configuration ---
DEFAULT_HORIZON = "monthly"  # or "yearly"
DEFAULT_FREQUENCY = "M15"

# NeuralForecast TFT parameters
TFT_HIDDEN_SIZE = 64
TFT_LAGS = 24
TFT_OUTPUT_SIZE = 1

# FinBERT sentiment configuration
FINBERT_MODEL = "ProsusAI/finbert"

# FRED TIPS Yields configuration
FRED_SERIES_TIPS = "TIPSYIELD"