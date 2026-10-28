"""Test run for mt5-agent modules - uses dynamic path resolution."""
import sys
from pathlib import Path

# Dynamically resolve the project root relative to this file's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "mt5-agent"))

from mt5_agent.gold_backtest_engine import fetch_gold_data
data = fetch_gold_data(period="30d", interval="1h")
print(f"Fetched {len(data)} rows of gold data OK")