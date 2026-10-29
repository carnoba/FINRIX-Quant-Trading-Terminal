"""Primary Modal GPU execution engine for macro XAUUSD forecasting.

Run:
    modal run macro_forecast_engine/modal_engine.py --mode monthly
    modal run macro_forecast_engine/modal_engine.py --mode yearly

All ingestion, FinBERT scoring, TFT training/inference, and Monte Carlo
analysis execute inside the A10G Modal function.
"""

from __future__ import annotations

import io
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import modal
except ImportError:  # pragma: no cover
    modal = None


APP_NAME = "xauusd-macro-quant-forecast"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "macro_month_forecast.json"

if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "neuralforecast",
            "pyfredapi",
            "transformers",
            "torch",
            "yfinance",
            "requests",
        )
    )
    app = modal.App(APP_NAME)
else:  # pragma: no cover
    image = None
    app = None


def _business_days_to_month_end(now: pd.Timestamp) -> int:
    last_day = now.normalize() + pd.offsets.MonthEnd(0)
    return max(1, len(pd.bdate_range(now.normalize(), last_day)))


def _normalise_ticker_frame(frame: pd.DataFrame, prefix: str) -> pd.Series:
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.index = pd.to_datetime(frame.index, utc=True).tz_localize(None)
    frame.columns = [str(column).lower() for column in frame.columns]
    return pd.to_numeric(frame["close"], errors="coerce").rename(prefix)


def _fetch_rss_titles_and_events(
    now: pd.Timestamp,
) -> tuple[list[str], list[dict[str, str]]]:
    import requests

    feeds = (
        "https://nfs.faireconomy.media/ff_calendar_thisweek.xml",
        "https://www.forexfactory.com/ff_calendar_thisweek.xml",
        "https://www.forexfactory.com/ff_news.xml",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    )
    keywords = ("nfp", "nonfarm", "cpi", "fomc", "rate decision", "interest rate")
    titles: list[str] = []
    events: list[dict[str, str]] = []
    for url in feeds:
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "macro-forecast-engine/1.0"},
                timeout=30,
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except (requests.RequestException, ET.ParseError):
            continue
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            if not title:
                continue
            titles.append(title)
            raw_date = item.findtext("pubDate") or item.findtext("date")
            parsed = pd.to_datetime(raw_date, errors="coerce", utc=True)
            if (
                not pd.isna(parsed)
                and parsed >= now
                and any(keyword in title.lower() for keyword in keywords)
            ):
                events.append(
                    {
                        "title": title,
                        "date": parsed.isoformat(),
                        "source": url,
                    }
                )
    unique_events = {
        (event["title"], event["date"]): event for event in events
    }
    return titles, sorted(unique_events.values(), key=lambda event: event["date"])


def _score_finbert(headlines: list[str]) -> float:
    if not headlines:
        return 0.0
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        top_k=None,
        device=0,
    )
    scores = []
    for output in classifier(headlines, truncation=True, max_length=256):
        values = {str(item["label"]).lower(): float(item["score"]) for item in output}
        scores.append(values.get("positive", 0.0) - values.get("negative", 0.0))
    return float(np.mean(scores)) if scores else 0.0


def _fetch_features(
    market_period: str,
    fred_api_key: str,
    now: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, str]]]:
    import requests
    import yfinance as yf

    gold = _normalise_ticker_frame(
        yf.download(
            "GC=F",
            period=market_period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        ),
        "gold_close",
    )
    dxy = _normalise_ticker_frame(
        yf.download(
            "DX-Y.NYB",
            period=market_period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        ),
        "dxy_close",
    )
    frame = pd.concat([gold, dxy], axis=1).sort_index().ffill().dropna()
    start = frame.index.min().date().isoformat()
    end = frame.index.max().date().isoformat()
    fred_params = {
        "id": "DFII10",
        "cosd": start,
        "coed": end,
    }
    if fred_api_key:
        fred_url = "https://api.stlouisfed.org/fred/series/observations"
        fred_params = {
            "series_id": "DFII10",
            "api_key": fred_api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
        }
    else:
        fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    fred_status = "DFII10 unavailable"
    try:
        response = requests.get(fred_url, params=fred_params, timeout=30)
        response.raise_for_status()
        if fred_api_key:
            observations = response.json().get("observations", [])
            tips_frame = pd.DataFrame(observations).rename(
                columns={"date": "date", "value": "tips10y"}
            )
            tips = tips_frame.set_index("date")["tips10y"]
        else:
            tips_frame = pd.read_csv(io.StringIO(response.text))
            date_column = (
                "DATE" if "DATE" in tips_frame.columns else "observation_date"
            )
            tips = tips_frame.set_index(date_column)["DFII10"]
        tips.index = pd.to_datetime(tips.index, utc=True).tz_localize(None)
        frame = frame.join(pd.to_numeric(tips, errors="coerce").rename("tips10y"))
        fred_status = "DFII10 loaded"
    except Exception as exc:
        frame["tips10y"] = np.nan
        fred_status = f"DFII10 unavailable: {type(exc).__name__}"

    headlines, events = _fetch_rss_titles_and_events(now)
    sentiment = _score_finbert(headlines)
    frame["tips10y"] = frame["tips10y"].ffill().bfill().fillna(0.0)
    frame["sentiment"] = sentiment
    frame["gold_return"] = frame["gold_close"].pct_change()
    frame["dxy_return"] = frame["dxy_close"].pct_change()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    ingestion = {
        "fred": fred_status,
        "finbert": "ProsusAI/finbert on A10G",
        "headline_count": len(headlines),
        "gpu_worker": "Modal A10G",
    }
    return frame, ingestion, events


def _fit_tft(features: pd.DataFrame, horizon: int) -> pd.Series:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TFT

    frame = features.copy()
    frame.index.name = "ds"
    frame = frame.reset_index()
    frame["unique_id"] = "XAUUSD"
    frame["y"] = frame["gold_close"].astype(float)
    exogenous = ["dxy_close", "tips10y", "sentiment"]
    train = frame[["unique_id", "ds", "y", *exogenous]].dropna()
    model = TFT(
        h=horizon,
        input_size=min(60, max(20, len(train) // 4)),
        max_steps=200,
        hist_exog_list=exogenous,
        val_check_steps=50,
        early_stop_patience_steps=3,
    )
    nf = NeuralForecast(models=[model], freq="D")
    validation_size = horizon if len(train) > horizon * 3 + 60 else 0
    nf.fit(df=train, val_size=validation_size)
    prediction = nf.predict()
    column = next(
        column for column in prediction.columns if column not in {"unique_id", "ds"}
    )
    business_dates = pd.bdate_range(
        features.index[-1].normalize() + pd.offsets.BDay(1),
        periods=horizon,
    )
    return pd.Series(
        prediction[column].to_numpy(dtype=float),
        index=business_dates,
        name="forecast",
    )


def _monte_carlo(
    last_price: float,
    forecast: pd.Series,
    historical_returns: pd.Series,
    simulations: int = 5000,
) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    returns = pd.to_numeric(historical_returns, errors="coerce").dropna()
    volatility = max(float(returns.tail(252).std()), 1e-5)
    base = forecast.to_numpy(dtype=float)
    shocks = rng.normal(
        0.0,
        volatility,
        size=(simulations, len(base)),
    )
    paths = base[None, :] * np.exp(np.cumsum(shocks, axis=1))
    probabilities = (paths > last_price).mean(axis=0)
    terminal = paths[:, -1]
    terminal_probability = float(probabilities[-1])
    bias = "bullish" if terminal_probability >= 0.5 else "bearish"
    confidence = terminal_probability if bias == "bullish" else 1.0 - terminal_probability

    shift = None
    if bias == "bearish":
        candidates = np.where(probabilities >= 0.60)[0]
        if len(candidates):
            start = int(candidates[0])
            end_candidates = np.where(probabilities[start:] < 0.60)[0]
            end = start + int(end_candidates[0] - 1) if len(end_candidates) else len(base) - 1
            shift = {
                "direction": "bullish",
                "start": pd.Timestamp(forecast.index[start]).date().isoformat(),
                "end": pd.Timestamp(forecast.index[end]).date().isoformat(),
                "confidence": round(float(probabilities[start] * 100), 2),
            }
    else:
        candidates = np.where(probabilities <= 0.40)[0]
        if len(candidates):
            start = int(candidates[0])
            end_candidates = np.where(probabilities[start:] > 0.40)[0]
            end = start + int(end_candidates[0] - 1) if len(end_candidates) else len(base) - 1
            shift = {
                "direction": "bearish",
                "start": pd.Timestamp(forecast.index[start]).date().isoformat(),
                "end": pd.Timestamp(forecast.index[end]).date().isoformat(),
                "confidence": round(float((1.0 - probabilities[start]) * 100), 2),
            }

    low, high = np.percentile(terminal, [5, 95])
    return {
        "target_price": round(float(np.median(terminal)), 4),
        "target_low": round(float(low), 4),
        "target_high": round(float(high), 4),
        "volatility": round(volatility, 8),
        "directional_bias": bias,
        "confidence_score": round(confidence * 100, 2),
        "simulations": simulations,
        "probability_bullish_by_day": [
            round(float(value), 6) for value in probabilities
        ],
        "simulation_targets": [round(float(value), 4) for value in terminal],
        "trend_shift_window": shift,
    }


def _event_metadata(events: list[dict[str, str]]) -> dict[str, Any]:
    if not events:
        return {
            "next_high_impact_title": "No upcoming CPI/NFP/FOMC event found",
            "next_high_impact_date": None,
            "recommended_rerun_date": "After the next high-impact event is published",
            "matched_events": [],
        }
    event = events[0]
    date = pd.Timestamp(event["date"])
    return {
        "next_high_impact_title": event["title"],
        "next_high_impact_date": date.isoformat(),
        "recommended_rerun_date": date.isoformat(),
        "matched_events": events[:10],
    }


def _run_worker(
    mode: str,
    market_period: str,
    fred_api_key: str,
) -> dict[str, Any]:
    if mode not in {"monthly", "yearly"}:
        raise ValueError("mode must be 'monthly' or 'yearly'")
    now = pd.Timestamp.now(tz="UTC")
    horizon = 252 if mode == "yearly" else _business_days_to_month_end(now)
    features, ingestion, events = _fetch_features(market_period, fred_api_key, now)
    forecast = _fit_tft(features, horizon)
    monte_carlo = _monte_carlo(
        float(features["gold_close"].iloc[-1]),
        forecast,
        features["gold_return"],
    )
    payload: dict[str, Any] = {
        "execution_environment": "Modal GPU A10G",
        "model": "NeuralForecast TFT",
        "mode": mode,
        "generated_at_utc": now.isoformat(),
        "data": {
            "start": features.index[0].isoformat(),
            "end": features.index[-1].isoformat(),
            "rows": len(features),
            "horizon": horizon,
            "horizon_unit": "business_days",
            "factors": ["GC=F", "DX-Y.NYB", "DFII10", "FinBERT sentiment"],
        },
        "forecast": [
            {"date": index.isoformat(), "price": round(float(value), 4)}
            for index, value in forecast.items()
        ],
        "monte_carlo": monte_carlo,
        "ingestion": ingestion,
        "economic_events": _event_metadata(events),
    }
    if mode == "yearly":
        payload["yearly_summary"] = {
            "target_price": monte_carlo["target_price"],
            "target_low": monte_carlo["target_low"],
            "target_high": monte_carlo["target_high"],
            "quarterly_pivots": {
                f"Q{quarter}": round(
                    float(forecast[forecast.index.quarter == quarter].iloc[-1]), 4
                )
                for quarter in range(1, 5)
                if not forecast[forecast.index.quarter == quarter].empty
            },
            "yearly_confidence_level": monte_carlo["confidence_score"],
        }
    return payload


def _print_summary(payload: dict[str, Any], output: Path) -> None:
    mc = payload["monte_carlo"]
    event = payload["economic_events"]
    shift = mc["trend_shift_window"]
    shift_text = "No threshold flip detected"
    if shift:
        shift_text = (
            f"Potential {shift['direction'].capitalize()} Reversal Window: "
            f"{shift['start']} - {shift['end']} "
            f"(Confidence: {shift['confidence']:.2f}%)"
        )
    print("\nMacro-Quant Modal Forecast")
    print("-" * 82)
    print(f"Execution Environment  : {payload['execution_environment']}")
    print(f"Mode / Horizon         : {payload['mode']} / {payload['data']['horizon']} business days")
    print(f"Target Price            : ${mc['target_price']:,.2f}")
    print(f"Volatility Range        : ${mc['target_low']:,.2f} - ${mc['target_high']:,.2f}")
    print(f"Confirmed Bias          : {mc['directional_bias'].capitalize()}")
    print(f"Confidence              : {mc['confidence_score']:.2f}%")
    print(f"Expected Trend Shift    : {shift_text}")
    print(f"Next High-Impact Event  : {event['next_high_impact_title']}")
    print(f"Next Event Date         : {event['next_high_impact_date'] or 'Unknown'}")
    print(f"Recommended Re-Run Date : {event['recommended_rerun_date']}")
    print(f"Output                  : {output}")


if modal is not None:

    @app.function(
        image=image,
        gpu="A10G",
        timeout=2400,
        retries=1,
        secrets=[modal.Secret.from_name("fred-secret")],
    )
    def forecast_worker(
        mode: str = "monthly",
        market_period: str = "10y",
        fred_api_key: str = "",
    ) -> dict[str, Any]:
        """Strict cloud-only pipeline: ingestion, TFT, and inference on A10G."""
        api_key = fred_api_key.strip() or os.getenv("FRED_API_KEY", "").strip()
        return _run_worker(mode, market_period, api_key)

    @app.local_entrypoint()
    def main(
        mode: str = "monthly",
        output: str = str(DEFAULT_OUTPUT),
        market_period: str = "10y",
    ) -> None:
        if mode not in {"monthly", "yearly"}:
            raise ValueError("--mode must be monthly or yearly")
        destination = Path(output)
        fred_api_key = os.getenv("FRED_API_KEY", "").strip()
        payload = forecast_worker.remote(
            mode=mode,
            market_period=market_period,
            fred_api_key=fred_api_key,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        _print_summary(payload, destination)
else:  # pragma: no cover
    raise RuntimeError("Modal is required. Run with: modal run modal_engine.py --mode monthly")
