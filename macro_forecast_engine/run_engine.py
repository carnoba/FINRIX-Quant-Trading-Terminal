"""Convenience launcher for the Modal-only XAUUSD forecast worker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .modal_engine import _print_summary, forecast_worker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "macro_month_forecast.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit the XAUUSD macro forecast to Modal A10G."
    )
    parser.add_argument("--mode", choices=("monthly", "yearly"), default="monthly")
    parser.add_argument("--market-period", default="10y")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    fred_api_key = os.getenv("FRED_API_KEY", "").strip()
    payload = forecast_worker.remote(
        mode=args.mode,
        market_period=args.market_period,
        fred_api_key=fred_api_key,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _print_summary(payload, args.output)


if __name__ == "__main__":
    main()
