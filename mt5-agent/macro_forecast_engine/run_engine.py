"""CLI for the macro-quant XAUUSD forecast pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .data_loader import (
        build_feature_frame,
        fetch_economic_events,
        fetch_live_news,
        fetch_market_data,
        fetch_tips_yield,
        load_news,
    )
    from .execution_layer import apply_macro_bias
    from .modal_engine import forecast_payload, train_and_forecast_worker
except ImportError:  # direct script execution
    from data_loader import (
        build_feature_frame,
        fetch_economic_events,
        fetch_live_news,
        fetch_market_data,
        fetch_tips_yield,
        load_news,
    )
    from execution_layer import apply_macro_bias
    from modal_engine import forecast_payload, train_and_forecast_worker


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "macro_month_forecast.json"


def trading_days_to_month_end(now: pd.Timestamp) -> int:
    last_day = now.normalize() + pd.offsets.MonthEnd(0)
    return max(1, len(pd.bdate_range(now.normalize(), last_day)))


def next_event_summary(events: list[dict]) -> tuple[str, str, str]:
    if not events:
        return "No upcoming NFP/CPI/FOMC event found", "unknown", "Re-run after the next high-impact event is published"
    event = events[0]
    timestamp = pd.Timestamp(event["date"])
    formatted = f"{timestamp.strftime('%b')} {timestamp.day}, {timestamp.year}"
    title = event["title"]
    return title, formatted, f"Re-run Model Post-Event ({formatted})"


def add_yearly_summary(payload: dict, features: pd.DataFrame) -> None:
    prices = pd.Series(
        [item["price"] for item in payload.get("forecast", [])],
        index=pd.to_datetime([item["date"] for item in payload.get("forecast", [])]),
    )
    if prices.empty:
        return
    pivots = {}
    for quarter in ("Q1", "Q2", "Q3", "Q4"):
        quarter_prices = prices[prices.index.quarter == int(quarter[1])]
        if not quarter_prices.empty:
            pivots[quarter] = round(float(quarter_prices.iloc[-1]), 2)
    payload["yearly_summary"] = {
        "target_price": round(float(prices.iloc[-1]), 2),
        "target_low": payload["month_end"]["target_low"],
        "target_high": payload["month_end"]["target_high"],
        "quarterly_pivots": pivots,
        "yearly_confidence_level": payload["month_end"]["confidence_score"],
    }


def load_local_m15(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(column).lower() for column in frame.columns]
    time_column = "time" if "time" in frame else "datetime"
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True).dt.tz_localize(None)
    return frame.set_index(time_column).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Macro-quant XAUUSD forecasting engine")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--local", action="store_true", help="Do not submit to Modal")
    parser.add_argument("--m15", type=Path, default=ROOT / "data" / "XAUUSD_M15.csv")
    parser.add_argument("--market-period", default="10y")
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--auto-month-end", action="store_true")
    parser.add_argument("--fresh-news", action="store_true")
    parser.add_argument("--yearly", action="store_true")
    args = parser.parse_args()

    now = pd.Timestamp.now(tz="UTC")
    if args.yearly:
        horizon = 252
    elif args.auto_month_end:
        horizon = trading_days_to_month_end(now)
    else:
        horizon = max(1, args.horizon)
    market = fetch_market_data(period=args.market_period, interval="1d")
    try:
        tips = fetch_tips_yield(
            start_date=market.index.min().date().isoformat(),
            end_date=market.index.max().date().isoformat(),
        )
    except Exception as exc:
        print(f"FRED DFII10 unavailable; continuing with neutral TIPS factor: {exc}")
        tips = None
    news = fetch_live_news() if args.fresh_news else load_news()
    features = build_feature_frame(market, tips=tips, news=news)
    features_json = features.to_json(orient="split", date_format="iso")
    if args.local or train_and_forecast_worker is None:
        payload = forecast_payload(features, horizon=horizon, simulations=5000)
    else:
        payload = train_and_forecast_worker.remote(features_json, horizon=horizon, simulations=5000)
    if args.m15.exists():
        m15 = load_local_m15(args.m15)
        close = m15["close"].astype(float)
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 21, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 21, adjust=False).mean()
        rsi = 100.0 - 100.0 / (1.0 + gain / loss.replace(0.0, pd.NA))
        raw_signal = pd.Series(0, index=m15.index, dtype=int)
        raw_signal.loc[rsi < 30] = 1
        raw_signal.loc[rsi > 70] = -1
        filtered = apply_macro_bias(raw_signal, payload)
        payload["execution_layer"] = {
            "strategy": "gold_backtest_engine RSI(21)",
            "macro_bias": payload["month_end"]["directional_bias"],
            "m15_rows": len(m15),
            "raw_nonzero_entries": int((raw_signal != 0).sum()),
            "filtered_signal_nonzero_entries": int((filtered != 0).sum()),
        }
    payload["data"] = {
        "start": features.index[0].isoformat(),
        "end": features.index[-1].isoformat(),
        "rows": len(features),
        "factors": ["GC=F", "DX-Y.NYB", "DFII10", "FinBERT sentiment"],
        "horizon": horizon,
        "horizon_mode": "yearly" if args.yearly else "auto_month_end" if args.auto_month_end else "fixed",
        "news_mode": "live_rss" if args.fresh_news else "local_cache",
        "news_headline_count": int(len(news)),
    }
    events = fetch_economic_events(now)
    event_title, event_date, rerun = next_event_summary(events)
    payload["economic_events"] = {
        "next_high_impact": event_title,
        "date": event_date,
        "recommendation": rerun,
        "matched_events": events[:10],
    }
    if args.yearly:
        add_yearly_summary(payload, features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    current_price = float(features["gold_close"].iloc[-1])
    month_end = payload["month_end"]
    bias = month_end["directional_bias"].capitalize()
    rule = f"Only M15 {'Buy' if bias == 'Bullish' else 'Sell' if bias == 'Bearish' else 'neutral'} Setups Active"
    print("\nMacro-Quant Forecast Summary")
    print("-" * 72)
    print(f"Run Date & Timestamp       : {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Current XAUUSD Price       : ${current_price:,.2f}")
    print(f"Forecast Target & Range    : ${month_end['month_end_target']:,.2f} "
          f"(${month_end['target_low']:,.2f} - ${month_end['target_high']:,.2f})")
    print(f"Confidence Score           : {month_end['confidence_score']:.2f}%")
    print(f"Active Directional Bias    : {bias}")
    print(f"Next High-Impact Event     : {event_title} ({event_date})")
    print(f"Recommended Model Re-run   : {rerun}")
    print(f"Execution Filter Rules     : {rule}")
    print(f"Forecast exported to       : {args.output}")


if __name__ == "__main__":
    main()
