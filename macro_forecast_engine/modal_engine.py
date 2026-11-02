"""Modal A10G institutional multi-horizon XAUUSD quantitative engine."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .config import DATA_DIR, FINBERT_MODEL, HORIZONS
    from .data_loader import (build_feature_frame, fetch_cot_data, fetch_live_news,
                              fetch_macro_data, fetch_market_data)
    from .trader_execution import build_execution_plan
    from .trader_risk import size_position
except ImportError:  # Modal may load this file as a top-level script.
    from macro_forecast_engine.config import DATA_DIR, FINBERT_MODEL, HORIZONS
    from macro_forecast_engine.data_loader import (
        build_feature_frame, fetch_cot_data, fetch_live_news,
        fetch_macro_data, fetch_market_data,
    )
    from macro_forecast_engine.trader_execution import build_execution_plan
    from macro_forecast_engine.trader_risk import size_position

try:
    import modal
except ImportError:  # pragma: no cover
    modal = None

APP_NAME = "xauusd-macro-quant-forecast"
if modal is not None:
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "neuralforecast==1.7.5", "transformers", "torch", "yfinance", "requests", "pandas", "numpy",
        "scikit-learn"
    ).add_local_python_source(
        "macro_forecast_engine"
    )
    app = modal.App(APP_NAME)
else:  # pragma: no cover
    image = app = None


def _business_dates(last: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.bdate_range(last.normalize() + pd.offsets.BDay(1), periods=horizon)


def _quantile_forecast(features: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Use TFT quantiles when supported, otherwise retain a calibrated drift baseline."""
    dates = _business_dates(features.index[-1], horizon)
    returns = features["gold_return"].dropna().tail(252)
    drift = float(returns.mean()) if not returns.empty else 0.0
    volatility = max(float(returns.std()), 0.002)
    prices = float(features["gold_close"].iloc[-1]) * np.exp(drift * np.arange(1, horizon + 1))
    result = pd.DataFrame(index=dates)
    result["q0.5"] = prices
    result["q0.1"] = prices * np.exp(-1.2816 * volatility * np.sqrt(np.arange(1, horizon + 1)))
    result["q0.9"] = prices * np.exp(1.2816 * volatility * np.sqrt(np.arange(1, horizon + 1)))
    diagnostics: dict[str, Any] = {
        "model_used": "calibrated_fallback",
        "train_loss": None,
        "valid_loss": None,
        "overfit_check": "not_available",
    }
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import TFT
        from neuralforecast.losses.pytorch import MQLoss
        from pytorch_lightning.callbacks import Callback, EarlyStopping
        import torch
        class MetricsCapture(Callback):
            def __init__(self) -> None:
                self.metrics: dict[str, Any] = {}
                self.global_step = 0

            def on_train_end(self, trainer: Any, pl_module: Any) -> None:
                self._capture(trainer)
                self.global_step = int(trainer.global_step)

            def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
                self._capture(trainer)

            def _capture(self, trainer: Any) -> None:
                metrics = dict(trainer.callback_metrics)
                metrics.update(trainer.logged_metrics)
                if metrics:
                    self.metrics = metrics

        metrics_capture = MetricsCapture()
        train = features.reset_index(names="ds")[["ds", "gold_close"]].rename(columns={"gold_close": "y"})
        train["unique_id"] = "XAUUSD"
        model = TFT(h=horizon, input_size=min(60, max(20, len(train) // 4)),
                    max_steps=500, val_check_steps=10, loss=MQLoss(level=[10, 50, 90]),
                    learning_rate=1e-4, dropout=0.30, optimizer_kwargs={"weight_decay": 1e-4},
                    optimizer=torch.optim.AdamW,
                    early_stop_patience_steps=15,
                    callbacks=[metrics_capture,
                               EarlyStopping(monitor="valid_loss", patience=15, mode="min",
                                             strict=False)])
        validation_size = horizon if len(train) > horizon * 3 else 0
        nf = NeuralForecast(models=[model], freq="D")
        nf.fit(train[["unique_id", "ds", "y"]], val_size=validation_size)
        metrics = metrics_capture.metrics
        def metric_value(name: str) -> float | None:
            value = metrics.get(name)
            return float(value.detach().cpu().item()) if value is not None and hasattr(value, "detach") else (
                float(value) if value is not None else None
            )
        def metric_fragment(fragment: str) -> float | None:
            for name in metrics:
                if fragment in str(name).lower():
                    value = metric_value(name)
                    if value is not None:
                        return value
            return None
        train_loss = next((metric_value(name) for name in (
            "train_loss_epoch", "train_loss", "ptl/train_loss", "train_loss_step"
        ) if metric_value(name) is not None), metric_fragment("train_loss"))
        valid_loss = next((metric_value(name) for name in (
            "valid_loss", "val_loss", "ptl/val_loss"
        ) if metric_value(name) is not None), metric_fragment("val_loss"))
        diagnostics = {
            "model_used": "NeuralForecast_TFT",
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "validation_size": validation_size,
            "early_stopping": {"patience": 15, "monitor": "ptl/val_loss"},
            "early_stopping_triggered": metrics_capture.global_step < 500,
            "overfit_check": (
                "validation_unavailable" if valid_loss is None else
                "review_required" if train_loss and valid_loss > train_loss * 3 else "within_tolerance"
            ),
        }
        prediction = nf.predict()
        columns = {str(column).lower(): column for column in prediction.columns}
        quantiles = [
            columns.get("tft-lo-90") or columns.get("tft-lo-10"),
            columns.get("tft-median") or columns.get("tft-lo-50"),
            columns.get("tft-hi-90") or columns.get("tft-hi-10"),
        ]
        if all(quantiles):
            ordered = np.sort(prediction[quantiles].to_numpy()[:horizon], axis=1)
            result["q0.1"] = ordered[:, 0]
            result["q0.5"] = ordered[:, 1]
            result["q0.9"] = ordered[:, 2]
    except (AttributeError, ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        diagnostics["fallback_reason"] = f"{type(error).__name__}: {error}"
    return result, diagnostics


def _monte_carlo(last_price: float, forecast: pd.DataFrame,
                 historical_returns: pd.Series, simulations: int = 100_000) -> dict[str, Any]:
    volatility = max(float(historical_returns.dropna().tail(252).std()), 1e-5)
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device=device).manual_seed(42)
        base = torch.tensor(forecast["q0.5"].to_numpy(), device=device, dtype=torch.float32)
        shocks = torch.randn((simulations, len(base)), generator=generator, device=device) * volatility
        paths = base.unsqueeze(0) * torch.exp(torch.cumsum(shocks, dim=1))
        terminal = paths[:, -1]
        probability = (paths > last_price).float().mean(dim=0).cpu().numpy()
        terminal_np = terminal.cpu().numpy()
    except (ImportError, RuntimeError):
        rng = np.random.default_rng(42)
        shocks = rng.normal(0.0, volatility, (simulations, len(forecast)))
        terminal_np = (forecast["q0.5"].to_numpy()[None, :] * np.exp(np.cumsum(shocks, axis=1)))[:, -1]
        probability = (terminal_np[:, None] > last_price).mean(axis=0)
    bullish_probability = float(probability[-1])
    bias = "bullish" if bullish_probability > 0.55 else "bearish" if bullish_probability < 0.45 else "neutral"
    confidence = bullish_probability if bias == "bullish" else 1 - bullish_probability if bias == "bearish" else 1 - abs(0.5 - bullish_probability) * 2
    low, high = np.percentile(terminal_np, [10, 90])
    return {
        "target_price": round(float(np.median(terminal_np)), 4),
        "target_low": round(float(low), 4), "target_high": round(float(high), 4),
        "volatility": round(volatility, 8), "directional_bias": bias,
        "confidence_score": round(confidence * 100, 2), "simulations": simulations,
        "probability_bullish_by_day": [round(float(value), 6) for value in probability],
        "drawdown_probability": round(float((terminal_np < last_price * 0.97).mean() * 100), 2),
    }


def _smc(features: pd.DataFrame) -> dict[str, Any]:
    recent = features.tail(30)
    low = float(recent["gold_close"].min())
    high = float(recent["gold_close"].max())
    return {
        "fair_value_gaps": [{"type": "bullish", "low": round(low, 4), "high": round(low * 1.002, 4)}],
        "liquidity_pools": {"bsl": round(high, 4), "ssl": round(low, 4)},
        "buy_zone": [round(low, 4), round(float(recent["gold_close"].quantile(0.35)), 4)],
        "sell_zone": [round(float(recent["gold_close"].quantile(0.65)), 4), round(high, 4)],
        "reaction_confidence": round(min(95.0, 50 + recent["gold_return"].abs().mean() * 1000), 2),
    }


def _run_worker(mode: str, market_period: str = "10y", fred_api_key: str = "") -> dict[str, Any]:
    if mode not in HORIZONS:
        raise ValueError(f"mode must be one of {', '.join(HORIZONS)}")
    if fred_api_key:
        os.environ["FRED_API_KEY"] = fred_api_key
    now = pd.Timestamp.now(tz="UTC")
    market = fetch_market_data(market_period)
    macro = fetch_macro_data(market.index.min().date().isoformat(), market.index.max().date().isoformat())
    cot = fetch_cot_data()
    news = fetch_live_news()
    features = build_feature_frame(market, macro, cot, news)
    forecast, training = _quantile_forecast(features, HORIZONS[mode])
    mc = _monte_carlo(float(features["gold_close"].iloc[-1]), forecast, features["gold_return"])
    smc = _smc(features)
    rows = []
    for index, row in forecast.iterrows():
        probability = mc["probability_bullish_by_day"][len(rows)]
        rows.append({"date": index.date().isoformat(), "target_q50": round(float(row["q0.5"]), 4),
                     "expected_low_q10": round(float(row["q0.1"]), 4),
                     "expected_high_q90": round(float(row["q0.9"]), 4),
                     "signal_bias": "bullish" if probability > 0.55 else "bearish" if probability < 0.45 else "neutral"})
    milestones = {f"Q{q}": rows[min(len(rows) - 1, q * len(rows) // 4 - 1)]["target_q50"]
                  for q in range(1, 5)} if mode == "yearly" else {
                      f"week_{q}": rows[min(len(rows) - 1, q * 5 - 1)]["target_q50"]
                      for q in range(1, max(2, len(rows) // 5 + 1))
                  }
    macro_bias = "neutral"
    if float(features["tips10y"].iloc[-1]) < float(features["tips10y"].tail(20).mean()):
        macro_bias = "bullish"
    elif float(features["tips10y"].iloc[-1]) > float(features["tips10y"].tail(20).mean()):
        macro_bias = "bearish"
    execution = build_execution_plan(float(features["gold_close"].iloc[-1]), rows, smc,
                                     mc["directional_bias"])
    risk = size_position(
        execution["entry_zone"][1] if execution["side"] == "long" else execution["entry_zone"][0],
        rows[0]["expected_low_q10"], rows[0]["target_q50"], rows[0]["expected_high_q90"],
        equity=100_000.0,
    )
    return {
        "execution_environment": "Modal GPU A10G",
        "model": "NeuralForecast TFT quantiles with calibrated fallback",
        "mode": mode, "generated_at_utc": now.isoformat(),
        "data": {"rows": len(features), "horizon": HORIZONS[mode], "horizon_unit": "business_days",
                 "factors": list(features.columns)},
        "training": training,
        "directional_bias": mc["directional_bias"], "confidence_score": mc["confidence_score"],
        "macro_directional_bias": macro_bias,
        "buy_zone": smc["buy_zone"], "sell_zone": smc["sell_zone"],
        "risk_levels": execution,
        "position_sizing": risk,
        "forecast": rows, "milestones": milestones, "monte_carlo": mc,
        "smc": smc, "ingestion": {"fred": ["DFII10", "DGS10-DGS2"], "cot": not cot.empty,
                                  "finbert": FINBERT_MODEL, "headline_count": len(news),
                                  "gpu_worker": "Modal A10G"},
    }


def _print_summary(payload: dict[str, Any], output: Path) -> None:
    print("\nInstitutional XAUUSD Quant Forecast")
    print("-" * 100)
    print(f"Selected Mode          : {payload['mode'].capitalize()}")
    print(f"Primary Directional Bias: {payload['directional_bias'].capitalize()} ({payload['confidence_score']:.2f}%)")
    print(f"Buy Zone               : {payload['buy_zone']}")
    print(f"Sell Zone              : {payload['sell_zone']}")
    print(f"Stop-Loss / TP1 / TP2  : {payload['risk_levels']}")
    print("Date         Target q50     Low q10       High q90      Signal")
    for row in payload["forecast"]:
        print(f"{row['date']}  {row['target_q50']:>12,.2f}  {row['expected_low_q10']:>12,.2f}  "
              f"{row['expected_high_q90']:>12,.2f}  {row['signal_bias']}")
    print(f"JSON Output             : {output}")


if modal is not None:
    @app.function(image=image, gpu="A10G", timeout=2400, retries=1,
                  secrets=[modal.Secret.from_name("fred-secret")])
    def forecast_worker(mode: str = "monthly", market_period: str = "10y",
                        fred_api_key: str = "") -> dict[str, Any]:
        return _run_worker(mode, market_period, fred_api_key or os.getenv("FRED_API_KEY", ""))

    @app.local_entrypoint()
    def main(mode: str = "monthly", output: str = "", market_period: str = "10y") -> None:
        filename_mode = {"monthly": "month", "yearly": "year"}.get(mode, mode)
        destination = Path(output) if output else DATA_DIR / f"macro_{filename_mode}_forecast.json"
        payload = forecast_worker.remote(mode=mode, market_period=market_period,
                                         fred_api_key=os.getenv("FRED_API_KEY", ""))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        _print_summary(payload, destination)
else:  # pragma: no cover
    raise RuntimeError("Modal is required. Run with: python -m modal run macro_forecast_engine/modal_engine.py")
