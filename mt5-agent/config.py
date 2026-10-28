"""Central configuration for the MT5 data + backtest pipeline.

Uses environment variables loaded via python-dotenv.
Sensitive values (broker credentials) should NEVER be hard-coded.
"""

import os
from pathlib import Path

# Base directory anchored to this file
BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# Broker / MT5 Connection Settings
# ------------------------------------------------------------
# BROKER_SERVER: Name of the MT5 server as appears in terminal
# (File -> Login to Trade Account). Examples: "Exness-Trial", "OctaFX-Demo"
BROKER_SERVER = os.getenv("BROKER_SERVER", "MetaQuotes-Demo")

# TERMINAL_PATH: Path to terminal64.exe if not registry-detected.
# Leave as None if MT5 is installed normally (registry-autodetect).
# Set only for portable installs, e.g.: r"C:\MT5Portable\terminal64.exe"
TERMINAL_PATH = os.getenv("MT5_TERMINAL_PATH")  # None by default

# MT5 Credentials — read from environment at runtime NEVER hardcode:
# MT5_LOGIN: Login ID (integer)
# MT5_PASSWORD: Password string
# These are read by dashboard.py via os.getenv("MT5_LOGIN") etc.
MT5_LOGIN = os.getenv("MT5_LOGIN")  # None until set at runtime
MT5_PASSWORD = os.getenv("MT5_PASSWORD")  # None until set at runtime

# ------------------------------------------------------------
# Data Settings
# ------------------------------------------------------------
SYMBOLS = ["EURUSD", "XAUUSD", "GBPUSD"]

TIMEFRAMES = {
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
}

HISTORY_START_YEAR = int(os.getenv("HISTORY_START_YEAR", "2020"))  # pull history from Jan 1 of this year
TICK_DAYS = int(os.getenv("TICK_DAYS", "60"))  # how many days of raw tick data to pull

DATA_DIR = str(BASE_DIR / "data")

# ------------------------------------------------------------
# Backtest Realism Settings
# ------------------------------------------------------------
# spread = constant bid-ask spread as a RELATIVE RATE of price
# (backtesting.py 0.6.x models fill at price * (1 +- spread))
# Read from env for flexibility: SPREAD=0.0002 or set SPREAD_env=0.0005
SPREAD = float(os.getenv("SPREAD", "0.0002"))

# Round-trip commission as a rate of trade value (0.0005 = 0.05%).
# Adjust to your broker (ECN raw spread + commission ≈ 0.0002-0.0008).
COMMISSION = float(os.getenv("COMMISSION", "0.0005"))

# Account leverage as margin ratio: 0.02 = 1:50 (typical retail max for majors),
# 0.01 = 1:100. Used only for margin-availability checks in the backtest.
MARGIN = float(os.getenv("MARGIN", "0.02"))

# Safety cap: never size beyond 40x effective leverage (stays under MARGIN cap).
MAX_LEVERAGE = int(os.getenv("MAX_LEVERAGE", "40"))

# ------------------------------------------------------------
# Backtest Capital Settings
# ------------------------------------------------------------
START_CASH = float(os.getenv("START_CASH", "10_000.0"))

# In-sample / out-of-sample split (0.7 = first 70% trains, last 30% validates)
OOS_SPLIT = float(os.getenv("OOS_SPLIT", "0.7"))