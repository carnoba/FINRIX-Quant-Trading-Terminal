"""backtest_hybrid.py
Thin entry point for the Hybrid SMC + S&R XAUUSD backtest
(HFM Cent Swap-Free environment). The full engine lives in
backtest_custom_hybrid.py; this wrapper just runs it.

Run:  python backtest_hybrid.py
"""
from backtest_custom_hybrid import finalize, main

if __name__ == "__main__":
    finalize(*main())
