"""Data loading layer: MT5 CSV -> pandas, with bundled-sample fallback so the
backtest pipeline can be validated before MT5 is even installed.

Priority:
  1. ./data/{SYMBOL}_{TF}.csv   (produced by data_pull.py)
  2. backtesting.test.EURUSD    (bundled sample - pipeline validation only)
"""
import os

import pandas as pd

import config


def load_ohlcv(symbol: str, tf: str, data_dir: str = config.DATA_DIR):
    """Load bars saved by data_pull.py. Returns (df, source_description)."""
    path = os.path.join(data_dir, f"{symbol}_{tf}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["time"], index_col="time")
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        # backtesting.py expects capitalized column names
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df = df.sort_index()
        if df[["Open", "High", "Low", "Close"]].isna().any().any():
            raise ValueError(f"NaN OHLC in {path}; re-run data_pull.py")
        return df, f"MT5 data: {path} ({len(df)} bars, {df.index[0]} -> {df.index[-1]})"

    print(f"[WARN] {path} not found -> falling back to bundled sample data.")
    print("       Run 'python data_pull.py' (with MT5 installed + logged in)")
    print("       to test on your broker's real history.")
    from backtesting.test import EURUSD

    df = EURUSD[["Open", "High", "Low", "Close", "Volume"]].copy()
    return df, f"bundled sample backtesting.test.EURUSD ({len(df)} bars, {df.index[0]} -> {df.index[-1]})"


def split_is_oos(df: pd.DataFrame, ratio: float = config.OOS_SPLIT):
    """Chronological in-sample / out-of-sample split (no shuffle = no leakage)."""
    cut = int(len(df) * ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
