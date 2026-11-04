# FINRIX XAUUSD Quant Trading Terminal

Cloud-native XAUUSD forecasting using a multi-horizon NeuralForecast TFT, FinBERT macro sentiment, FRED real yields and curve spreads, CFTC positioning, ETF flows, and yfinance market data. Training, inference, sentiment scoring, Monte Carlo simulation, and SMC overlays run inside a Modal A10G GPU worker.

## Architecture

- **Market factors:** Gold futures, DXY, GVZ, GLD, and IAU daily prices/volume.
- **Macro factors:** FRED `DFII10` real yields and `DGS10-DGS2` curve spread.
- **Positioning/news:** CFTC gold commercial/non-commercial net positioning and GPU FinBERT sentiment velocity.
- **Forecast model:** NeuralForecast Temporal Fusion Transformer with q10/q50/q90 outputs.
- **Risk analysis:** 100,000-path CUDA Monte Carlo corridors, drawdown probability, dynamic risk levels, and SMC zones.
- **Execution:** Modal GPU function with output persisted to `data/macro_{mode}_forecast.json`.

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

Daily forecast (30 business days):

```bash
python -m modal run macro_forecast_engine/modal_engine.py --mode daily
```

Monthly forecast (22 business days):

```bash
python -m modal run macro_forecast_engine/modal_engine.py --mode monthly
```

Yearly forecast (252 business days):

```bash
python -m modal run macro_forecast_engine/modal_engine.py --mode yearly
```

### Interactive terminal application

Launch the green Rich/questionary terminal:

```bash
python app.py
```

On Windows, add the repository directory to `PATH` once, then launch directly:

```cmd
finrix
```

FINRIX creates and persists `data/finrix_config.json`. The System Settings menu
controls Modal A10G versus local CPU execution, default account equity, and
default risk per trade. Forecast actions use those saved values automatically.

The menu can submit live Daily, Monthly, and Yearly Modal GPU forecasts, inspect
FinBERT news sentiment, calculate dynamic SMC/TFT execution levels and Kelly
sizing, replay historical strategies without lookahead, and display this guide.

## Output

Each mode writes `data/macro_{mode}_forecast.json`. The JSON contains:

```json
{
  "execution_environment": "Modal GPU A10G",
  "mode": "daily",
  "data": {"horizon": 30, "horizon_unit": "business_days"},
  "monte_carlo": {
    "target_price": 4500.0,
    "target_low": 4200.0,
    "target_high": 4800.0,
    "drawdown_probability": 12.4
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
