"""
fetch_news_events.py — Lightweight high-impact news scraper for XAUUSD (Gold/USD).

Pulls today's high-impact economic events that move Gold (USD) from ForexFactory.
Saves results to mt5-agent/data/news_events_today.json for consumption by the
Streamlit dashboard and execution pipeline.

Usage:
    python mt5-agent/fetch_news_events.py

Requires: requests, beautifulsoup4
    pip install requests beautifulsoup4
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "news_events_today.json"

FOREX_FACTORY_URL = "https://www.forexfactory.com/calendar"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

IMPACT_MAP = {
    "high": "High",
    "medium": "Medium",
    "low": "Low",
    "non-economic": "Non-Economic",
    "holiday": "Holiday",
}

GOLD_USD_KEYWORDS = [
    "nfp", "non-farm", "nonfarm", "unemployment", "jobs",
    "cpi", "inflation", "pce", "ppi", "gdp", "retail sales",
    "fomc", "fed", "federal reserve", "powell", "interest rate",
    "rate decision", "monetary policy", "average hourly earnings",
    "core cpi", "core pce", "advance gdp", "gdp price index",
    "initial jobless claims", "continuing claims",
    "ism", "services", "manufacturing", "pm",
    "durable goods", "trade balance", "current account",
]


def _impact_from_row(row) -> str | None:
    """Extract impact level from a ForexFactory calendar row."""
    impact_cell = row.find("td", class_=re.compile(r"impact"))
    if not impact_cell:
        span = row.find("span", class_=re.compile(r"impact"))
        if span:
            cls = span.get("class", [])
            for c in cls:
                c_lower = c.lower()
                for key, val in IMPACT_MAP.items():
                    if key in c_lower:
                        return val
    else:
        spans = impact_cell.find_all("span")
        for sp in spans:
            cls = sp.get("class", [])
            for c in cls:
                c_lower = c.lower()
                for key, val in IMPACT_MAP.items():
                    if key in c_lower:
                        return val
        title = impact_cell.get("title", "").lower()
        for key, val in IMPACT_MAP.items():
            if key in title:
                return val
    return None


def _text(cell) -> str:
    return cell.get_text(" ", strip=True) if cell else ""


def _is_high_impact(impact: str | None) -> bool:
    return impact is not None and impact.lower() == "high"


def _is_gold_usd_relevant(event_text: str) -> bool:
    """Return True if the event is likely to move XAUUSD / USD."""
    t = event_text.lower()
    return any(kw in t for kw in GOLD_USD_KEYWORDS)


def fetch_forexfactory() -> list[dict]:
    """Scrape today's high-impact USD events from ForexFactory calendar."""
    try:
        resp = requests.get(FOREX_FACTORY_URL, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        print(f"[fetch_news] ForexFactory request failed: {exc}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    events: list[dict] = []

    rows = soup.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 5:
            continue

        time_cell = cells[0]
        currency_cell = cells[1] if len(cells) > 1 else None
        impact_cell = cells[2] if len(cells) > 2 else None
        event_cell = cells[3] if len(cells) > 3 else None

        event_time = _text(time_cell)
        currency = _text(currency_cell)
        impact = _impact_from_row(impact_cell) if impact_cell else None
        event_name = _text(event_cell)

        if not event_name or not currency:
            continue

        if currency.upper() != "USD":
            continue
        if not _is_high_impact(impact):
            continue
        if not _is_gold_usd_relevant(event_name):
            continue

        events.append({
            "time": event_time or "TBD",
            "currency": currency,
            "impact": impact,
            "event": event_name,
            "source": "ForexFactory",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        })

    return events


def save_events(events: list[dict], path: Path = OUTPUT_FILE) -> None:
    payload = {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "source": "ForexFactory",
        "count": len(events),
        "events": events,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[fetch_news] Saved {len(events)} high-impact USD events to {path}")


def main() -> None:
    print("[fetch_news] Fetching high-impact USD news from ForexFactory...")
    events = fetch_forexfactory()
    if events:
        save_events(events)
        for ev in events:
            print(f"  [{ev['time']}] {ev['impact']} — {ev['event']}")
    else:
        print("[fetch_news] No high-impact USD events found today (or scrape blocked).")
        payload = {
            "date": datetime.now(timezone.utc).date().isoformat(),
            "source": "ForexFactory",
            "count": 0,
            "events": [],
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        OUTPUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()