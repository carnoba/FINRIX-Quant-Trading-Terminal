"""Institutional cross-asset and macro data ingestion helpers."""

from __future__ import annotations

import os
import re
from io import StringIO
import xml.etree.ElementTree as ET
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import RobustScaler

try:
    from .data_processor import add_sentiment_and_flow
except ImportError:  # Modal may load this file as a top-level script.
    from macro_forecast_engine.data_processor import add_sentiment_and_flow

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


def _download(ticker: str, period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is required for market data ingestion")
    frame = yf.download(ticker, period=period, interval=interval, auto_adjust=False,
                        progress=False, threads=False)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if frame.empty:
        raise ValueError(f"No data returned for {ticker}")
    frame.index = pd.to_datetime(frame.index, utc=True).tz_localize(None)
    frame.columns = [str(column).lower() for column in frame.columns]
    return frame


def fetch_market_data(period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    """Fetch gold, DXY, GVZ, ETF holdings/volume, and align daily factors."""
    frames: list[pd.DataFrame] = []
    for ticker, prefix in (("GC=F", "gold"), ("DX-Y.NYB", "dxy"),
                           ("^GVZ", "gvz"), ("GLD", "gld"), ("IAU", "iau")):
        try:
            data = _download(ticker, period, interval)
        except (RuntimeError, ValueError, requests.RequestException):
            continue
        columns = {"close": f"{prefix}_close", "volume": f"{prefix}_volume"}
        available = {source: target for source, target in columns.items() if source in data.columns}
        frames.append(data.rename(columns=available)[list(available.values())])
    if not frames or "gold_close" not in frames[0]:
        raise ValueError("Gold market data is unavailable")
    frame = frames[0]
    for other in frames[1:]:
        frame = frame.join(other, how="outer")
    return frame.sort_index().ffill().dropna(subset=["gold_close"])


def _fred_series(series_id: str, start_date: Optional[str], end_date: Optional[str]) -> pd.Series:
    params = {"id": series_id, "cosd": start_date or "", "coed": end_date or ""}
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if api_key:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {"series_id": series_id, "api_key": api_key, "file_type": "json",
                  "observation_start": start_date or "", "observation_end": end_date or ""}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = pd.DataFrame(response.json().get("observations", []))
        series = data.set_index("date")["value"]
    else:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = pd.read_csv(StringIO(response.text))
        date_column = "DATE" if "DATE" in data.columns else "observation_date"
        series = data.set_index(date_column)[series_id].replace(".", np.nan)
    series.index = pd.to_datetime(series.index, utc=True).tz_localize(None)
    return pd.to_numeric(series, errors="coerce").rename(series_id).dropna()


def fetch_macro_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch TIPS real yields and the 10Y-2Y Treasury curve spread."""
    result = pd.DataFrame(index=pd.date_range(start_date, end_date, freq="D"))
    for series_id, name in (("DFII10", "tips10y"), ("DGS10", "ust10y"), ("DGS2", "ust2y")):
        try:
            result = result.join(_fred_series(series_id, start_date, end_date), how="left")
        except (requests.RequestException, ValueError, KeyError):
            result[name] = np.nan
    if "DGS10" in result and "DGS2" in result:
        result["curve_spread_10y2y"] = result["DGS10"] - result["DGS2"]
    else:
        result["curve_spread_10y2y"] = np.nan
    return result.rename(columns={"DFII10": "tips10y"}).drop(columns=["DGS10", "DGS2"], errors="ignore")


def fetch_cot_data() -> pd.DataFrame:
    """Fetch current CFTC disaggregated gold positioning when available."""
    url = "https://publicreporting.cftc.gov/resource/jun7-fc8e.json"
    try:
        response = requests.get(url, params={"$limit": 2000, "$q": "GOLD"}, timeout=30)
        response.raise_for_status()
        rows = response.json()
        data = pd.DataFrame(rows)
        date_col = next((c for c in ("report_date_as_yyyy_mm_dd", "report_date") if c in data), None)
        if date_col is None:
            return pd.DataFrame()
        data.index = pd.to_datetime(data[date_col], errors="coerce")
        result = pd.DataFrame(index=data.index)
        for prefix, target in (("comm_positions", "commercial_positions"),
                               ("noncomm_positions", "noncommercial_positions")):
            long = next((c for c in data.columns if c.startswith(f"{prefix}_long")), None)
            short = next((c for c in data.columns if c.startswith(f"{prefix}_short")), None)
            if long and short:
                result[target] = (
                    pd.to_numeric(data[long], errors="coerce")
                    - pd.to_numeric(data[short], errors="coerce")
                )
        return result.groupby(level=0).last().sort_index()
    except (requests.RequestException, ValueError, KeyError):
        return pd.DataFrame()


GOLD_NEWS_FEEDS: tuple[tuple[str, str], ...] = (
    ("forexlive_gold", "https://www.forexlive.com/feed/news"),
    ("investing_commodities", "https://www.investing.com/rss/commodities.rss"),
    ("marketwatch_macro", "https://feeds.marketwatch.com/marketwatch/marketpulse/"),
)
GOLD_MACRO_KEYWORDS = re.compile(
    r"\b("
    r"gold|xauusd|fomc rate|cpi|inflation|non[- ]farm payrolls?|"
    r"dxy|yields?"
    r")\b",
    re.IGNORECASE,
)


def filter_gold_macro_headlines(headlines: Iterable[Any]) -> list[str]:
    """Keep only headlines tied to gold macro drivers."""
    return [text for item in headlines if (text := str(item).strip())
            and GOLD_MACRO_KEYWORDS.search(text)]


def fetch_live_news() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source, feed_url in GOLD_NEWS_FEEDS:
        try:
            response = requests.get(feed_url, headers={"User-Agent": "macro-forecast-engine/1.0"}, timeout=30)
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except (requests.RequestException, ET.ParseError):
            continue
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            date = pd.to_datetime(item.findtext("pubDate"), errors="coerce", utc=True)
            if title and not pd.isna(date) and GOLD_MACRO_KEYWORDS.search(title):
                rows.append({"date": date.tz_localize(None), "headline": title, "source": source})
    result = pd.DataFrame(rows, columns=["date", "headline", "source"])
    if result.empty:
        return result
    return result.drop_duplicates(subset=["headline"]).sort_values("date", ascending=False).reset_index(drop=True)


def score_headlines(headlines: Iterable[Any], model_name: str = "ProsusAI/finbert") -> float:
    texts = filter_gold_macro_headlines(headlines)
    if not texts:
        return float("nan")
    try:
        from transformers import pipeline
        classifier = pipeline("text-classification", model=model_name, top_k=None, device=0)
        values = []
        for output in classifier(texts, truncation=True, max_length=256):
            labels = output if isinstance(output, list) else [output]
            scores = {str(item["label"]).lower(): float(item["score"]) for item in labels}
            positive = scores.get("positive", scores.get("bullish", 0.0))
            negative = scores.get("negative", scores.get("bearish", 0.0))
            neutral = scores.get("neutral", 0.0)
            values.append((positive - negative) * (1.0 - 0.25 * neutral))
        return float(np.mean(values)) if values else float("nan")
    except (ImportError, OSError, RuntimeError, ValueError):
        bullish = re.compile(
            r"\b(weak dollar|dollar falls?|dxy falls?|lower yield|rate cut|"
            r"dovish|recession|uncertainty|gold rises?)\b", re.I,
        )
        bearish = re.compile(
            r"\b(strong dollar|dollar rises?|dxy rises?|higher yield|yields? rise|"
            r"rate hike|hawkish|hot inflation|cpi rises|ppi rises|payrolls beat)\b", re.I,
        )
        values = []
        for text in texts:
            positive = len(bullish.findall(text))
            negative = len(bearish.findall(text))
            values.append((positive - negative) / max(1, positive + negative))
        return float(np.mean(values)) if values else float("nan")


def build_feature_frame(daily: pd.DataFrame, macro: Optional[pd.DataFrame] = None,
                        cot: Optional[pd.DataFrame] = None,
                        news: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Align market, macro, positioning, ETF, volatility, and NLP velocity features."""
    frame = daily.copy().sort_index()
    if macro is not None:
        frame = frame.join(macro.reindex(frame.index, method="ffill"), how="left")
    if cot is not None and not cot.empty:
        frame = frame.join(cot.reindex(frame.index, method="ffill"), how="left")
    sentiment = 0.0
    if news is not None and not news.empty:
        scored_sentiment = score_headlines(news["headline"].tolist())
        sentiment = 0.0 if pd.isna(scored_sentiment) else float(scored_sentiment)
    frame = add_sentiment_and_flow(frame, sentiment, cot)
    frame["sentiment"] = sentiment
    frame["sentiment_velocity"] = frame["sentiment"].diff().fillna(0.0)
    for column in ("tips10y", "curve_spread_10y2y", "commercial_positions",
                   "noncommercial_positions", "gvz_close", "gld_volume", "iau_volume"):
        if column not in frame:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").ffill().bfill().fillna(0.0)
    scaled_columns = [
        "gvz_close", "gld_volume", "iau_volume",
        "commercial_positions", "noncommercial_positions",
    ]
    frame[scaled_columns] = RobustScaler().fit_transform(frame[scaled_columns])
    frame["gold_return"] = frame["gold_close"].pct_change()
    frame["dxy_return"] = frame["dxy_close"].pct_change()
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["gold_close"])
