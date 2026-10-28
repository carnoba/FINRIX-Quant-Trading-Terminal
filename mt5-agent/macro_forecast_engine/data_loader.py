"""Market, macroeconomic, and financial-news ingestion."""

from __future__ import annotations

import json
import os
from io import StringIO
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

try:
    import pyfredapi as pf
except ImportError:  # pragma: no cover
    pf = None

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover
    pipeline = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEWS = ROOT / "data" / "news_events_today.json"


def _download(ticker: str, period: str, interval: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is required for market data ingestion")
    frame = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if frame.empty:
        raise ValueError(f"No data returned for {ticker}")
    frame.index = pd.to_datetime(frame.index, utc=True).tz_localize(None)
    frame.columns = [str(column).lower() for column in frame.columns]
    return frame


def fetch_market_data(period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    """Fetch GC=F, DXY, and align daily market factors."""
    gold = _download("GC=F", period, interval).add_prefix("gold_")
    dxy = _download("DX-Y.NYB", period, interval).add_prefix("dxy_")
    frame = gold.join(dxy, how="outer").sort_index().ffill().dropna(
        subset=["gold_close", "dxy_close"]
    )
    return frame


def fetch_tips_yield(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.Series:
    """Fetch the 10-year inflation-indexed Treasury yield (FRED DFII10)."""
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if api_key:
        if pf is None:
            raise RuntimeError("pyfredapi is required when FRED_API_KEY is set")
        kwargs = {"api_key": api_key}
        if start_date:
            kwargs["observation_start"] = start_date
        if end_date:
            kwargs["observation_end"] = end_date
        result = pf.get_series(series_id="DFII10", **kwargs)
        if isinstance(result, pd.Series):
            series = result
        elif isinstance(result, pd.DataFrame):
            value_column = "value" if "value" in result.columns else result.columns[-1]
            date_column = "date" if "date" in result.columns else result.columns[0]
            series = result.set_index(date_column)[value_column]
        else:
            raise TypeError("Unexpected pyfredapi response for DFII10")
        series.index = pd.to_datetime(series.index, utc=True).tz_localize(None)
        return pd.to_numeric(series, errors="coerce").rename("tips10y").dropna()

    params = {"id": "DFII10", "cosd": start_date or "", "coed": end_date or ""}
    response = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    csv = pd.read_csv(StringIO(response.text))
    date_column = "DATE" if "DATE" in csv.columns else "observation_date"
    if date_column not in csv.columns or "DFII10" not in csv.columns:
        raise ValueError("FRED public CSV did not contain a date and DFII10 columns")
    series = csv.set_index(date_column)["DFII10"].replace(".", np.nan)
    series.index = pd.to_datetime(series.index, utc=True).tz_localize(None)
    return pd.to_numeric(series, errors="coerce").rename("tips10y").dropna()


def _headline_texts(headlines: Iterable[Any]) -> list[str]:
    texts = []
    for item in headlines:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, Mapping):
            text = item.get("title") or item.get("headline") or item.get("text")
            if text:
                texts.append(str(text))
    return texts


def score_headlines(
    headlines: Iterable[Any],
    model_name: str = "ProsusAI/finbert",
) -> float:
    """Return mean FinBERT positive-minus-negative sentiment."""
    texts = _headline_texts(headlines)
    if not texts:
        return 0.0
    if pipeline is None:
        return 0.0
    classifier = pipeline("text-classification", model=model_name, top_k=None)
    scores = []
    for output in classifier(texts, truncation=True, max_length=256):
        labels = {str(item["label"]).lower(): float(item["score"]) for item in output}
        scores.append(labels.get("positive", 0.0) - labels.get("negative", 0.0))
    return float(np.mean(scores)) if scores else 0.0


def load_news(path: Path | str = DEFAULT_NEWS) -> pd.DataFrame:
    """Load local news JSON; callers can replace this with an API adapter."""
    source = Path(path)
    if not source.exists():
        return pd.DataFrame(columns=["date", "headline", "sentiment"])
    payload = json.loads(source.read_text(encoding="utf-8"))
    events = payload.get("events", payload if isinstance(payload, list) else [])
    rows = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        headline = event.get("title") or event.get("headline") or event.get("event")
        date = event.get("date") or event.get("datetime") or payload.get("date")
        if headline and date:
            rows.append({"date": pd.to_datetime(date), "headline": str(headline)})
    news = pd.DataFrame(rows)
    if news.empty:
        return pd.DataFrame(columns=["date", "headline", "sentiment"])
    news["sentiment"] = score_headlines(news["headline"].tolist())
    return news


def fetch_live_news() -> pd.DataFrame:
    """Fetch ForexFactory RSS, falling back to CNBC RSS, then score headlines."""
    feeds = (
        "https://www.forexfactory.com/ff_news.xml",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    )
    rows = []
    for feed_url in feeds:
        try:
            response = requests.get(
                feed_url,
                headers={"User-Agent": "macro-forecast-engine/1.0"},
                timeout=30,
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
            for item in root.findall(".//item"):
                title = item.findtext("title")
                timestamp = item.findtext("pubDate")
                if title:
                    rows.append(
                        {
                            "date": pd.to_datetime(timestamp, errors="coerce", utc=True),
                            "headline": title.strip(),
                        }
                    )
            if rows:
                break
        except (requests.RequestException, ET.ParseError):
            continue
    news = pd.DataFrame(rows)
    if news.empty:
        return pd.DataFrame(columns=["date", "headline", "sentiment"])
    news = news.dropna(subset=["date"])
    news["date"] = news["date"].dt.tz_localize(None)
    news["sentiment"] = [
        score_headlines([headline]) for headline in news["headline"]
    ]
    return news


def fetch_economic_events(now: Optional[pd.Timestamp] = None) -> list[dict[str, Any]]:
    """Fetch upcoming high-impact events from calendar/news RSS feeds."""
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    feeds = (
        "https://nfs.faireconomy.media/ff_calendar_thisweek.xml",
        "https://www.forexfactory.com/ff_calendar_thisweek.xml",
        "https://www.forexfactory.com/ff_news.xml",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    )
    keywords = ("nfp", "nonfarm", "cpi", "fomc", "rate decision", "interest rate")
    events: list[dict[str, Any]] = []
    for feed_url in feeds:
        try:
            response = requests.get(
                feed_url,
                headers={"User-Agent": "macro-forecast-engine/1.0"},
                timeout=30,
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except (requests.RequestException, ET.ParseError):
            continue
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            if not title or not any(word in title.lower() for word in keywords):
                continue
            raw_date = item.findtext("pubDate") or item.findtext("date")
            event_date = pd.to_datetime(raw_date, errors="coerce", utc=True)
            if pd.isna(event_date):
                continue
            if event_date >= current:
                events.append(
                    {
                        "title": title,
                        "date": event_date.isoformat(),
                        "source": feed_url,
                    }
                )
    unique = { (event["title"], event["date"]): event for event in events }
    return sorted(unique.values(), key=lambda event: event["date"])


def build_feature_frame(
    daily: pd.DataFrame,
    tips: Optional[pd.Series] = None,
    news: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Create the daily multi-factor frame consumed by NeuralForecast."""
    frame = daily.copy().sort_index()
    if tips is not None:
        frame = frame.join(tips.resample("1D").last(), how="left")
    else:
        frame["tips10y"] = np.nan
    if news is not None and not news.empty:
        news_daily = news.set_index("date")["sentiment"].resample("1D").mean()
        frame = frame.join(news_daily.rename("sentiment"), how="left")
    else:
        frame["sentiment"] = 0.0
    frame["tips10y"] = frame["tips10y"].ffill().bfill()
    frame["sentiment"] = frame["sentiment"].fillna(0.0)
    frame["gold_return"] = frame["gold_close"].pct_change()
    frame["dxy_return"] = frame["dxy_close"].pct_change()
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["gold_close"])


def align_to_m15(daily_features: pd.DataFrame, m15: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill daily macro factors onto an M15 price index."""
    macro = daily_features.reindex(m15.index, method="ffill")
    return m15.join(macro.drop(columns=["gold_open", "gold_high", "gold_low", "gold_volume"], errors="ignore"), how="left")
