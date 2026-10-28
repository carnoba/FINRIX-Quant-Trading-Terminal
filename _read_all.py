"""Read all files in the mt5-agent directory - uses dynamic path resolution."""
from pathlib import Path

# Dynamically resolve the mt5-agent directory relative to this file's location
SCRIPT_DIR = Path(__file__).resolve().parent.parent / "mt5-agent"

# Iterate over Python files in mt5-agent
py_files = sorted(SCRIPT_DIR.glob("*.py"))
for f in py_files:
    if f.name.startswith("_") or f.name in {"gold_backtest_engine.py"}:
        continue  # Skip internal/private files
    print(f"  {f.name}: {f.stat().st_size} bytes, modified {f.stat().st_mtime:.0f}")