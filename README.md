# Institutional XAUUSD Modal Macro Forecasting Engine

Cloud-native XAUUSD forecasting using NeuralForecast TFT, FinBERT financial-news sentiment, FRED DFII10 real yields, and yfinance market data. Training, inference, sentiment scoring, Monte Carlo simulation, and event detection run inside a Modal A10G GPU worker.

## Architecture

- **Market factors:** Gold futures (`GC=F`) and DXY (`DX-Y.NYB`) from yfinance.
- **Macro factor:** FRED `DFII10`, the 10-year inflation-indexed Treasury real yield.
- **News factor:** ForexFactory/CNBC RSS headlines scored with `ProsusAI/finbert`.
- **Forecast model:** NeuralForecast Temporal Fusion Transformer.
- **Risk analysis:** 5,000-path Monte Carlo target ranges, confidence, directional bias, and trend-shift windows.
- **Execution:** Modal GPU function with output persisted to `data/macro_month_forecast.json`.

## Quickstart Execution Guide

### Step 1: Clone and install

```bash
git clone https://github.com/carnoba/trading-prediction.git
cd trading-prediction
pip install -r requirements.txt
```

### Step 2: Configure Modal and FRED

```bash
modal setup
modal secret create fred-secret FRED_API_KEY=your_key
```

The worker also supports the public FRED CSV endpoint when no key is supplied. Never commit real credentials.

### Step 3: Run a forecast

Monthly forecast:

```bash
python -m modal run macro_forecast_engine/modal_engine.py --mode monthly
```

Yearly forecast:

```bash
python -m modal run macro_forecast_engine/modal_engine.py --mode yearly
```

## Output

Both modes write `data/macro_month_forecast.json`. The JSON contains:

```json
{
  "execution_environment": "Modal GPU A10G",
  "mode": "monthly",
  "data": {"horizon": 3, "horizon_unit": "business_days"},
  "monte_carlo": {
    "target_price": 4500.0,
    "target_low": 4200.0,
    "target_high": 4800.0,
    "directional_bias": "bullish",
    "confidence_score": 66.5,
    "trend_shift_window": null
  },
  "economic_events": {
    "next_high_impact_title": "FOMC Rate Decision",
    "next_high_impact_date": "2026-11-05T...",
    "recommended_rerun_date": "2026-11-05T..."
  }
}
```

The terminal summary reports **Target Price**, **Volatility Range**, **Confirmed Bias**, **Confidence**, **Expected Trend Shift**, **Next High-Impact Event**, and **Recommended Re-Run Date**.

## Repository Layout

```text
macro_forecast_engine/
├── modal_engine.py       # Primary Modal A10G pipeline
├── run_engine.py         # Forecast orchestration utilities
├── data_loader.py        # Market, FRED, RSS, and sentiment ingestion
├── execution_layer.py    # Macro-bias signal filtering utilities
├── config.py             # Environment-backed configuration
└── __init__.py
requirements.txt
.env.example
```

## License

MIT License. See [LICENSE](LICENSE).
