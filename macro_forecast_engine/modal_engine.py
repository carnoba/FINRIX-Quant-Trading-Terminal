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
    from .trader_timemachine import (
        STRATEGIES, TIMEFRAME_FETCH, normalize_timeframe_prices, run_strategy_matrix,
        strategy_confidence,
    )
except ImportError:  # Modal may load this file as a top-level script.
    from macro_forecast_engine.config import DATA_DIR, FINBERT_MODEL, HORIZONS
    from macro_forecast_engine.data_loader import (
        build_feature_frame, fetch_cot_data, fetch_live_news,
        fetch_macro_data, fetch_market_data,
    )
    from macro_forecast_engine.trader_execution import build_execution_plan
    from macro_forecast_engine.trader_risk import size_position
    from macro_forecast_engine.trader_timemachine import (
        STRATEGIES, TIMEFRAME_FETCH, normalize_timeframe_prices, run_strategy_matrix,
        strategy_confidence,
    )

try:
    import modal
except ImportError:  # pragma: no cover
    modal = None

APP_NAME = "xauusd-macro-quant-forecast"
MIN_GOLD_PRICE = 2500.0
MAX_GOLD_PRICE = 6500.0
MAX_ANNUALIZED_VOLATILITY = 0.35
VOLATILITY_MEAN_REVERSION = 0.92
if modal is not None:
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "neuralforecast==1.7.5", "transformers", "torch", "yfinance", "requests", "pandas", "numpy",
        "scikit-learn", "rich", "questionary"
    ).add_local_python_source(
        "macro_forecast_engine"
    )
    app = modal.App(APP_NAME)
else:  # pragma: no cover
    image = app = None


def _business_dates(last: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.bdate_range(last.normalize() + pd.offsets.BDay(1), periods=horizon)


def _returns_to_prices(last_price: float, returns: np.ndarray) -> np.ndarray:
    """Compound predicted log-returns back into nominal price levels."""
    return float(last_price) * np.exp(np.cumsum(np.asarray(returns, dtype=float), axis=0))


def _quantile_forecast(features: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Forecast log-returns, then invert them into nominal price quantiles."""
    dates = _business_dates(features.index[-1], horizon)
    last_price = float(features["gold_close"].iloc[-1])
    returns = np.log1p(features["gold_return"].dropna()).tail(252)
    drift = float(returns.mean()) if not returns.empty else 0.0
    volatility = max(float(returns.std()), 0.002)
    if horizon >= 252:
        volatility = min(volatility, MAX_ANNUALIZED_VOLATILITY / np.sqrt(252.0))
    steps = np.arange(1, horizon + 1)
    median_returns = np.full(horizon, drift)
    prices = _returns_to_prices(last_price, median_returns)
    result = pd.DataFrame(index=dates)
    result["q0.5"] = prices
    result["q0.1"] = last_price * np.exp(drift * steps - 1.2816 * volatility * np.sqrt(steps))
    result["q0.9"] = last_price * np.exp(drift * steps + 1.2816 * volatility * np.sqrt(steps))
    floor = max(MIN_GOLD_PRICE, last_price * 0.55)
    ceiling = min(MAX_GOLD_PRICE, last_price * 1.60)
    result["q0.1"] = result["q0.1"].clip(lower=floor, upper=ceiling)
    result["q0.5"] = result["q0.5"].clip(lower=floor, upper=ceiling)
    result["q0.9"] = result["q0.9"].clip(lower=floor, upper=ceiling)
    result["q0.1"] = np.minimum(result["q0.1"], result["q0.5"])
    result["q0.9"] = np.maximum(result["q0.9"], result["q0.5"])
    diagnostics: dict[str, Any] = {
        "model_used": "calibrated_fallback",
        "train_loss": None,
        "valid_loss": None,
        "overfit_check": "not_available",
        "target_transform": "log_return",
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
                self.validation_losses: list[float] = []
                self.training_losses: list[float] = []

            def on_train_end(self, trainer: Any, pl_module: Any) -> None:
                self._capture(trainer)
                self.global_step = int(trainer.global_step)

            def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
                self._capture(trainer)

            def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
                self._capture(trainer)
                if self.validation_losses:
                    self.metrics["valid_loss"] = float(np.mean(self.validation_losses))
                    self.validation_losses.clear()

            def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
                self._capture(trainer)
                if self.training_losses:
                    self.metrics["train_loss_epoch"] = float(np.mean(self.training_losses))

            def on_validation_batch_end(
                self, trainer: Any, pl_module: Any, outputs: Any, batch: Any,
                batch_idx: int, dataloader_idx: int = 0,
            ) -> None:
                value = self._scalar(outputs)
                if value is not None:
                    self.validation_losses.append(value)

            def on_train_batch_end(
                self, trainer: Any, pl_module: Any, outputs: Any, batch: Any,
                batch_idx: int,
            ) -> None:
                value = self._scalar(outputs)
                if value is not None:
                    self.training_losses.append(value)

            def _capture(self, trainer: Any) -> None:
                metrics = dict(trainer.callback_metrics)
                metrics.update(trainer.logged_metrics)
                if metrics:
                    self.metrics = metrics

            @staticmethod
            def _scalar(value: Any) -> float | None:
                if isinstance(value, dict):
                    value = value.get("loss", value.get("valid_loss"))
                if hasattr(value, "detach"):
                    value = value.detach().cpu().item()
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

        metrics_capture = MetricsCapture()
        train = features.reset_index(names="ds")[["ds", "gold_close"]].copy()
        train["y"] = np.log(train["gold_close"]).diff().fillna(0.0)
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
        train_trajectories = getattr(nf.models[0], "train_trajectories", [])
        if train_trajectories:
            train_loss = float(train_trajectories[-1][1])
        trajectories = getattr(nf.models[0], "valid_trajectories", [])
        if valid_loss is None and trajectories:
            valid_loss = float(trajectories[-1][1])
        if train_loss is None and metrics_capture.training_losses:
            train_loss = float(np.mean(metrics_capture.training_losses))
        diagnostics = {
            "model_used": "NeuralForecast_TFT",
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "validation_size": validation_size,
            "target_transform": "log_return",
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
            ordered_returns = np.sort(prediction[quantiles].to_numpy()[:horizon], axis=1)
            result["q0.1"] = _returns_to_prices(last_price, ordered_returns[:, 0])
            result["q0.5"] = _returns_to_prices(last_price, ordered_returns[:, 1])
            result["q0.9"] = _returns_to_prices(last_price, ordered_returns[:, 2])
    except (AttributeError, ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        diagnostics["fallback_reason"] = f"{type(error).__name__}: {error}"
    result["q0.1"] = result["q0.1"].clip(lower=floor, upper=ceiling)
    result["q0.5"] = result["q0.5"].clip(lower=floor, upper=ceiling)
    result["q0.9"] = result["q0.9"].clip(lower=floor, upper=ceiling)
    result["q0.1"] = np.minimum(result["q0.1"], result["q0.5"])
    result["q0.9"] = np.maximum(result["q0.9"], result["q0.5"])
    return result, diagnostics


def _monte_carlo(last_price: float, forecast: pd.DataFrame,
                 historical_returns: pd.Series, simulations: int = 100_000) -> dict[str, Any]:
    volatility = max(float(historical_returns.dropna().tail(252).std()), 1e-5)
    horizon = len(forecast)
    annualized_volatility = min(volatility * np.sqrt(252.0), MAX_ANNUALIZED_VOLATILITY)
    step_volatility = annualized_volatility / np.sqrt(252.0)
    floor = max(MIN_GOLD_PRICE, last_price * 0.55)
    ceiling = min(MAX_GOLD_PRICE, last_price * 1.60)
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device=device).manual_seed(42)
        base = torch.tensor(forecast["q0.5"].to_numpy(), device=device, dtype=torch.float32)
        shocks = torch.randn((simulations, horizon), generator=generator, device=device) * step_volatility
        deviations = torch.zeros_like(shocks)
        for step in range(horizon):
            previous = deviations[:, step - 1] if step else 0.0
            deviations[:, step] = VOLATILITY_MEAN_REVERSION * previous + shocks[:, step]
        paths = base.unsqueeze(0) * torch.exp(deviations)
        paths = torch.clamp(paths, min=floor, max=ceiling)
        terminal = paths[:, -1]
        probability = (paths > last_price).float().mean(dim=0).cpu().numpy()
        terminal_np = terminal.cpu().numpy()
    except (ImportError, RuntimeError):
        rng = np.random.default_rng(42)
        shocks = rng.normal(0.0, step_volatility, (simulations, horizon))
        deviations = np.zeros_like(shocks)
        for step in range(horizon):
            previous = deviations[:, step - 1] if step else 0.0
            deviations[:, step] = VOLATILITY_MEAN_REVERSION * previous + shocks[:, step]
        paths = forecast["q0.5"].to_numpy()[None, :] * np.exp(deviations)
        paths = np.clip(paths, floor, ceiling)
        terminal_np = paths[:, -1]
        probability = (paths > last_price).mean(axis=0)
    bullish_probability = float(probability[-1])
    bias = "bullish" if bullish_probability > 0.55 else "bearish" if bullish_probability < 0.45 else "neutral"
    confidence = bullish_probability if bias == "bullish" else 1 - bullish_probability if bias == "bearish" else 1 - abs(0.5 - bullish_probability) * 2
    low, high = np.percentile(terminal_np, [10, 90])
    return {
        "target_price": round(float(np.median(terminal_np)), 4),
        "target_low": round(float(low), 4), "target_high": round(float(high), 4),
        "volatility": round(annualized_volatility, 8), "directional_bias": bias,
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


def _run_worker(
    mode: str,
    market_period: str = "10y",
    fred_api_key: str = "",
    account_equity: float = 100_000.0,
    risk_per_trade_pct: float = 1.0,
    local_trend: bool = False,
) -> dict[str, Any]:
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
    local_bias = None
    local_confidence = None
    if local_trend:
        log_returns = np.log1p(features["gold_return"].dropna()).tail(120)
        weighted_drift = float(log_returns.ewm(span=20, adjust=False).mean().iloc[-1])
        if abs(weighted_drift) < 1e-7:
            log_prices = np.log(features["gold_close"].tail(60))
            weighted_drift = float(log_prices.diff().ewm(span=20, adjust=False).mean().iloc[-1])
        weighted_drift = float(np.clip(weighted_drift, -0.003, 0.003))
        local_bias = "bullish" if weighted_drift > 0 else "bearish" if weighted_drift < 0 else "neutral"
        local_confidence = round(float(np.clip(
            50.0 + abs(weighted_drift) * 8000.0, 50.0, 74.0,
        )), 2)
        last_price = float(features["gold_close"].iloc[-1])
        steps = np.arange(1, len(forecast) + 1)
        trend_median = last_price * np.exp(weighted_drift * steps)
        local_volatility = max(float(log_returns.std()), 0.002)
        forecast["q0.5"] = trend_median
        forecast["q0.1"] = trend_median * np.exp(-1.2816 * local_volatility * np.sqrt(steps))
        forecast["q0.9"] = trend_median * np.exp(1.2816 * local_volatility * np.sqrt(steps))
    mc = _monte_carlo(float(features["gold_close"].iloc[-1]), forecast, features["gold_return"])
    if local_trend and local_bias is not None and local_confidence is not None:
        mc["directional_bias"] = local_bias
        mc["confidence_score"] = local_confidence
    smc = _smc(features)
    rows = []
    for index, row in forecast.iterrows():
        probability = mc["probability_bullish_by_day"][len(rows)]
        signal_bias = (
            local_bias if local_trend and local_bias is not None
            else "bullish" if probability > 0.55 else "bearish" if probability < 0.45 else "neutral"
        )
        rows.append({"date": index.date().isoformat(), "target_q50": round(float(row["q0.5"]), 4),
                     "expected_low_q10": round(float(row["q0.1"]), 4),
                     "expected_high_q90": round(float(row["q0.9"]), 4),
                     "signal_bias": signal_bias})
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
        equity=account_equity,
        risk_fraction=max(0.0, float(risk_per_trade_pct)) / 100.0,
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
        "risk_configuration": {
            "account_equity": float(account_equity),
            "risk_per_trade_pct": float(risk_per_trade_pct),
        },
        "forecast": rows, "milestones": milestones, "monte_carlo": mc,
        "smc": smc, "ingestion": {"fred": ["DFII10", "DGS10-DGS2"], "cot": not cot.empty,
                                  "finbert": FINBERT_MODEL, "headline_count": len(news),
                                  "gpu_worker": "Modal A10G"},
    }


def _print_summary(payload: dict[str, Any], output: Path) -> None:
    print("\nInstitutional XAUUSD Quant Forecast")
    print("-" * 100)
    training = payload["training"]
    risk = payload["risk_levels"]
    mc = payload["monte_carlo"]
    print(f"Mode & Horizon Days    : {payload['mode'].capitalize()} / {payload['data']['horizon']}")
    print(f"Target Price ($)       : {mc['target_price']:,.4f}")
    print(f"Volatility Corridor ($): {mc['target_low']:,.4f} - {mc['target_high']:,.4f}")
    print(f"Directional Bias       : {payload['directional_bias'].capitalize()} "
          f"({payload['confidence_score']:.2f}%)")
    print(f"SMC Buy Reaction Zone  : {payload['buy_zone']}")
    print(f"SMC Sell Reaction Zone : {payload['sell_zone']}")
    print(f"Kelly Position Size    : {payload['position_sizing']['units']:.6f} units/lots")
    print(f"Dynamic SL             : {risk['stop_loss']:,.4f}")
    print(f"Dynamic TP1 / TP2 / TP3: {risk['take_profit_1']:,.4f} / "
          f"{risk['take_profit_2']:,.4f} / {risk['take_profit_3']:,.4f}")
    print(f"Model Losses           : train={training.get('train_loss')} | "
          f"valid={training.get('valid_loss')}")
    print("Date         Target q50     Low q10       High q90      Signal")
    for row in payload["forecast"]:
        print(f"{row['date']}  {row['target_q50']:>12,.2f}  {row['expected_low_q10']:>12,.2f}  "
              f"{row['expected_high_q90']:>12,.2f}  {row['signal_bias']}")
    print(f"Generated JSON Artifact : {output}")


if modal is not None:
    @app.function(image=image, gpu="A10G", timeout=2400, retries=1,
                  secrets=[modal.Secret.from_name("fred-secret")])
    def forecast_worker(
        mode: str = "monthly",
        market_period: str = "10y",
        fred_api_key: str = "",
        account_equity: float = 100_000.0,
        risk_per_trade_pct: float = 1.0,
    ) -> dict[str, Any]:
        return _run_worker(
            mode,
            market_period,
            fred_api_key or os.getenv("FRED_API_KEY", ""),
            account_equity,
            risk_per_trade_pct,
        )

    @app.function(image=image, gpu="A10G", timeout=1800, retries=1)
    def strategy_backtest_worker(strategy: str, timeframe: str,
                                 market_period: str = "5y") -> dict[str, Any]:
        default_period, interval = TIMEFRAME_FETCH[timeframe]
        period = default_period if timeframe != "1D" else market_period
        market = fetch_market_data(period, interval=interval)
        prices = normalize_timeframe_prices(market["gold_close"], timeframe)
        result = run_strategy_matrix(prices, [strategy], [timeframe])
        row = result.iloc[0].to_dict()
        row["confidence"] = strategy_confidence(row)
        return row

    @app.local_entrypoint()
    def main(
        mode: str = "monthly",
        output: str = "",
        market_period: str = "10y",
        account_equity: float = 100_000.0,
        risk_per_trade_pct: float = 1.0,
    ) -> None:
        if mode == "ensemble":
            destination = Path(output) if output else DATA_DIR / "modal_ensemble.json"
            tasks = [(strategy, timeframe) for timeframe in ("1D", "4h", "1h")
                     for strategy in STRATEGIES]
            rows = list(strategy_backtest_worker.map(
                [strategy for strategy, _ in tasks],
                [timeframe for _, timeframe in tasks],
                [market_period] * len(tasks),
            ))
            rows.sort(key=lambda row: (float(row["confidence"]),
                                       float(row["net_expected_return"])), reverse=True)
            payload = {
                "execution_environment": "Modal GPU A10G",
                "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "leaderboard": rows[:5],
                "recommendation": f"{rows[0]['strategy']} / {rows[0]['timeframe']}",
            }
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
            print(f"Modal A10G ensemble artifact: {destination}")
            for rank, row in enumerate(rows[:5], 1):
                print(f"{rank}. {row['strategy']} / {row['timeframe']} | "
                      f"win={row['win_rate']:.2f}% confidence={row['confidence']:.2f}%")
            return
        filename_mode = {"monthly": "month", "yearly": "year"}.get(mode, mode)
        destination = Path(output) if output else DATA_DIR / f"macro_{filename_mode}_forecast.json"
        payload = forecast_worker.remote(mode=mode, market_period=market_period,
                                         fred_api_key=os.getenv("FRED_API_KEY", ""),
                                         account_equity=account_equity,
                                         risk_per_trade_pct=risk_per_trade_pct)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        _print_summary(payload, destination)
else:  # pragma: no cover
    raise RuntimeError("Modal is required. Run with: python -m modal run macro_forecast_engine/modal_engine.py")
