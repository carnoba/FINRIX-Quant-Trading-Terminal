# FINRIX XAUUSD Quant Trading Terminal

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100-rc1/)
[![Modal GPU Sync](https://img.shields.io/badge/Modal-GPU%20A10G-orange.svg)](https://modal.com/)
[![Zero-Trust Audit PASSED](https://img.shields.io/badge/Audit-PASSED-brightgreen.svg)](FINRIX_SECURITY_AND_QA_AUDIT_REPORT.md)
[![14-Strategy Engine](https://img.shields.io/badge/Strategies-14-brightgreen.svg)](macro_forecast_engine/)

## One-Line Execution

Run the full FINRIX terminal directly from command line:

```bash
finrix
```

Or launch the Rich/questionary terminal:

```bash
python app.py
```

The System Settings menu controls Modal A10G vs local CPU, default equity, and risk per trade -- all forecast actions use saved values automatically.

## Core Architecture & Key Features

### Dual-Path Execution Engine

- **Local CPU** -- Lightweight drift & baseline execution, runs instantly on any machine
- **Modal A10G Cloud GPU** -- NeuralForecast TFT deep learning, 100,000-path CUDA Monte Carlo, CUDA-accelerated quantile regression

### Quantile Forecasting Envelope

Monotonic price corridors $q_{10} \le q_{50} \le q_{90}$ with dynamic volatility decay across daily (30 business days), monthly (22 business days), and yearly (252 business days) horizons.

### Dynamic Equity-Based Position Sizing

Real-time dollar risk math with pip-based stop-loss/take-profit calculation, equity compounding, and fractional-Kelly allocation across 14 strategies.

### 14-Strategy Backtesting & Parameter Engine

All strategies persist to `data/finrix_indicator_params.json`:

| Strategy | Indicators |
|----------|------------|
| SMC Fair Value Gap (FVG) | FVG detection, order block breakout |
| SMC Liquidity Order Block Breakout | SSL/BSL liquidity pool detection |
| EMA Golden Cross / Death Cross | Fast/slow EMA crossover, trend period |
| Triple EMA Ribbon | Fast/medium/slow EMA ribbon dynamics |
| RSI Mean Reversion | RSI period, overbought, oversold |
| MACD Zero-Line & Signal Line Crossover | Fast/slow/signal MACD components |
| Bollinger Bands Squeeze & Reversal | Period, standard deviation |
| Supertrend ATR Trend Follower | ATR period, multiplier |
| Stochastic Oscillator | K period, D period, slowing |
| ADX + DI Trend Classifier | ADX period, DI threshold |
| Ichimoku Cloud Kumo Breakout | Tenkan, Kijun, Senkou B periods |
| ATR Volatility Expansion Channel | ATR period |
| TFT Log-Return Trend | Temporal Fusion Transformer ML |
| Volatility Parity & Risk-Adjusted Allocation | Volatility-normalized allocation |

### Macro Sentiment Analysis

Live RSS headlines processed via **FinBERT** NLP model with dovish/hawkish net score cancellation math. Fallback regex-based scoring when transformer models are unavailable.

## Terminal Menu Workflow Guide (Options 1-8)

| Option | Description |
|--------|-------------|
| **1** | Instant Execution Card -- Kelly Position Sizing, dynamic Entry bounds, Dynamic SL (exact pip distance), Risk:Reward ratios |
| **2** | Daily Price Path & Signals -- 30-day forward forecast with quantile monotonicity ($q_{10} \le q_{50} \le q_{90}$) |
| **3** | Monthly Macro Horizon -- 22-day outlook & weekly pivots with dynamic bias calculation |
| **4** | Yearly Regime Trend Shift -- 252-day forecast with volatility envelopes, $q_{90}$ capped at 6500 |
| **5** | Macro News & FinBERT Sentiment -- Live RSS headline aggregation, Dovish/Hawkish score cancellation |
| **6** | Risk & Position Sizing Calculator -- Custom market price overrides, pip distance math, zero static hardcoded outputs |
| **7** | Historical Strategy Backtester -- All 14 strategies across 15m, 1h, 4h, 1D timeframes, dynamic lot sizing |
| **8** | Multi-Strategy Ensemble Analyzer -- Top 5 leaderboard, dynamic header title (Local CPU vs Modal A10G), all strategies |

### System Settings

- Toggle execution environment (local/modal)
- Set default account equity
- Set default risk per trade %
- Customize indicator parameters (all 10 groups with limits)
- Save & apply settings persistence via `data/finrix_config.json`

## Future Roadmap

- **Multi-asset expansion** -- Forex pairs, Crypto indices, Equity indices
- **Direct MetaTrader 5 (MT5) execution bridge** & Webhook integration
- **Automated portfolio rebalancing** with dynamic risk limits
- **Walking-forward analysis** & rolling-origin backtesting
- **Cluster deployment** across multiple Modal GPU workers

## Repository Layout

```text
macro_forecast_engine/
├── modal_engine.py       # Primary Modal A10G pipeline
├── run_engine.py         # Forecast orchestration utilities
├── data_loader.py        # Market, FRED, RSS, and sentiment ingestion
├── execution_layer.py    # Macro-bias signal filtering utilities
├── config.py             # Environment-backed configuration
├── indicator_params.py   # Persistent indicator parameter JSON
├── finrix_config.py      # Terminal settings persistence
├── trader_execution.py   # SMC + TFT execution levels
├── trader_risk.py        # Kelly sizing & volatility parity
├── trader_timemachine.py # Strategy replay & confidence scoring
├── config.py
├── __init__.py
requirements.txt
.env.example
README.md (this file)
FINRIX_SECURITY_AND_QA_AUDIT_REPORT.md
data/
├── finrix_config.json
├── finrix_indicator_params.json
├── macro_daily_forecast.json
├── macro_month_forecast.json
└── macro_year_forecast.json
cache/
gold_market_cache.csv
```

--- 

MIT License. See [LICENSE](LICENSE).
