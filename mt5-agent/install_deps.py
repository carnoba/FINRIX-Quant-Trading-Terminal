#!/usr/bin/env python
import subprocess
import sys

pkgs = ['vectorbt', 'pandas_ta', 'yfinance', 'scikit-learn']
for pkg in pkgs:
    try:
        __import__(pkg)
        print(f"{pkg}: already installed")
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])
        print(f"{pkg}: installed successfully")
print("All dependencies resolved")