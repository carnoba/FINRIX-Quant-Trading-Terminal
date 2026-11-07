"""Local CPU or Modal forecast launcher."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .modal_engine import _print_summary, _run_worker, forecast_worker
from .indicator_params import load_indicator_params


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_FILENAMES = {
    "daily": "macro_daily_forecast.json",
    "monthly": "macro_month_forecast.json",
    "yearly": "macro_year_forecast.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the XAUUSD forecast engine.")
    parser.add_argument("--mode", choices=tuple(ARTIFACT_FILENAMES), default="monthly")
    parser.add_argument("--market-period", default="10y")
    parser.add_argument("--environment", choices=("local", "modal"), default="local")
    parser.add_argument("--account-equity", type=float, default=1000.0)
    parser.add_argument("--risk-per-trade-pct", type=float, default=1.0)
    args = parser.parse_args()

    fred_api_key = os.getenv("FRED_API_KEY", "").strip()
    if args.environment == "local":
        indicator_params = load_indicator_params()
        payload = _run_worker(
            args.mode,
            args.market_period,
            fred_api_key,
            account_equity=args.account_equity,
            risk_per_trade_pct=args.risk_per_trade_pct,
            local_trend=True,
        )
        payload["execution_environment"] = "Local CPU"
        payload["confidence_score"] = round(float(np.clip(
            payload.get("confidence_score", 50.0), 50.0, 74.0,
        )), 2)
        payload["indicator_parameters"] = indicator_params
    else:
        payload = forecast_worker.remote(
            mode=args.mode,
            market_period=args.market_period,
            fred_api_key=fred_api_key,
        )
    # Resolve the artifact from a fixed allowlist rather than a user-supplied
    # path, so the write target can never escape the project data directory.
    output = PROJECT_ROOT / "data" / ARTIFACT_FILENAMES[args.mode]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _print_summary(payload, output)


if __name__ == "__main__":
    main()
