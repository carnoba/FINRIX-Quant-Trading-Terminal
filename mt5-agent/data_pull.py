"""Step 1a: Pull historical OHLCV bars and raw ticks from MT5 (READ-ONLY).

- Never places orders; attaches to the terminal session (UI login by you).
- CLI args allow small chunked pulls (fits slow shells / first-time downloads):
    python -u data_pull.py --symbols EURUSD --tfs H1 H4 --years 3 --tick-days 10
- Saves CSVs under config.DATA_DIR so backtests are reproducible without MT5.
"""
import argparse
import os
import sys
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd

import config

sys.stdout.reconfigure(line_buffering=True)  # unbuffered logs (flush=True everywhere)

MT5_TIMEFRAMES = {
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
}


def connect() -> bool:
    kwargs = {}
    if config.TERMINAL_PATH:
        kwargs["path"] = config.TERMINAL_PATH
    if not mt5.initialize(**kwargs):
        code, msg = mt5.last_error()
        print(f"[ERROR] MT5 terminal not found (code {code}: {msg}).")
        return False
    return True


def pull_bars(symbol: str, tf_name: str, date_from: datetime):
    """Pull bars; on failure retry from progressively later start dates,
    then fall back to MONTHLY chunked pulls (demo servers often refuse
    large M5 range requests)."""
    span_start = date_from
    for attempt in range(4):
        rates = mt5.copy_rates_range(symbol, MT5_TIMEFRAMES[tf_name], span_start, datetime.now())
        if rates is not None and len(rates) > 0:
            return _rates_to_df(rates)
        code, msg = mt5.last_error()
        print(f"  {symbol} {tf_name}: attempt {attempt + 1} failed ({code} {msg})", flush=True)
        span_start = pd.Timestamp(span_start) + pd.DateOffset(years=1)
        if span_start >= pd.Timestamp(datetime.now()):
            break
    # monthly chunked fallback: walk backwards from now, stop at first failure
    chunks = []
    end = pd.Timestamp(datetime.now())
    while end > pd.Timestamp(date_from):
        start = max(end - pd.DateOffset(months=1), pd.Timestamp(date_from))
        rates = mt5.copy_rates_range(symbol, MT5_TIMEFRAMES[tf_name], start, end)
        if rates is None or len(rates) == 0:
            print(f"  {symbol} {tf_name}: chunk {start.date()}..{end.date()} "
                  f"failed {mt5.last_error()} - stopping", flush=True)
            break
        chunks.append(pd.DataFrame(rates))
        end = start
    if not chunks:
        return None
    df = pd.concat(chunks[::-1], ignore_index=True)
    return _rates_to_df(df.to_records(index=False))


def _rates_to_df(rates) -> pd.DataFrame:
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")  # MT5 stores UTC seconds
    df = df.drop_duplicates("time").set_index("time").sort_index()
    return df[["open", "high", "low", "close", "tick_volume"]].rename(
        columns={"tick_volume": "volume"})


def pull_ticks(symbol: str, date_from: datetime):
    ticks = mt5.copy_ticks_range(symbol, date_from, datetime.now(), mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        print(f"  {symbol} ticks: FAILED {mt5.last_error()}", flush=True)
        return None
    df = pd.DataFrame(ticks)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.set_index("time").sort_index()


def quality_report(df: pd.DataFrame, tf_name: str | None = None) -> None:
    msg = f"  rows={len(df)} span={df.index[0]} -> {df.index[-1]}"
    if tf_name == "H1" and len(df) > 2:
        expected = pd.date_range(df.index[0], df.index[-1], freq="1h")
        msg += f" | missing H1 bars: {len(expected) - len(df)} (weekends/holidays)"
    ohlc = [c for c in ("open", "high", "low", "close") if c in df.columns]
    if len(ohlc) == 4 and df[ohlc].isna().any().any():
        msg += " | WARNING: NaN OHLC present"
    print(msg, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=config.SYMBOLS)
    ap.add_argument("--tfs", nargs="+", default=list(config.TIMEFRAMES))
    ap.add_argument("--years", type=int, default=None,
                    help="history length in years (default: since HISTORY_START_YEAR)")
    ap.add_argument("--tick-days", type=int, default=config.TICK_DAYS)
    args = ap.parse_args()

    if not connect():
        return 2
    acc = mt5.account_info()
    print(f"Attached: {acc.login}@{acc.server} balance={acc.balance}", flush=True)

    date_from = (datetime.now() - pd.DateOffset(years=args.years)) if args.years \
        else datetime(config.HISTORY_START_YEAR, 1, 1)
    tick_from = datetime.now() - pd.Timedelta(days=args.tick_days)
    os.makedirs(config.DATA_DIR, exist_ok=True)

    saved = failed = 0
    for symbol in args.symbols:
        print(f"{symbol}:", flush=True)
        if not mt5.symbol_select(symbol, True):
            print(f"  symbol not available: {mt5.last_error()}", flush=True)
            continue
        for tf_name in args.tfs:
            df = pull_bars(symbol, tf_name, date_from)
            if df is not None:
                df.to_csv(os.path.join(config.DATA_DIR, f"{symbol}_{tf_name}.csv"))
                quality_report(df, tf_name)
                saved += 1
            else:
                failed += 1
        ticks = pull_ticks(symbol, tick_from) if args.tick_days > 0 else None
        if ticks is not None:
            path = os.path.join(config.DATA_DIR, f"{symbol}_TICKS_{args.tick_days}d.csv")
            ticks.to_csv(path)
            quality_report(ticks)
            saved += 1
        elif args.tick_days > 0:
            failed += 1

    mt5.shutdown()
    print(f"Done: saved={saved} failed={failed} -> {config.DATA_DIR}", flush=True)
    return 0 if saved else 1


if __name__ == "__main__":
    sys.exit(main())
