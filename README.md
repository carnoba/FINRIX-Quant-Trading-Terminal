# FINRIX XAUUSD Quant Trading Terminal

> **Status: Research Prototype / Quantitative Framework.** FINRIX supports
> reproducible market research, forecasting, and backtesting. It is not
> investment advice and does not execute live orders without a separately
> installed and explicitly authorized broker bridge.

[![CI Status](https://img.shields.io/github/actions/workflow/status/carnoba/trading-prediction/.github/workflows/ci.yml?label=CI&logo=github&color=blue)](https://github.com/carnoba/trading-prediction/actions/workflows/ci.yml)
[![Pytest Pass](https://img.shields.io/badge/pytest-207%20passed-brightgreen.svg)](https://github.com/carnoba/trading-prediction/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100-rc1/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Execution: Local CPU | Modal A10G GPU](https://img.shields.io/badge/Execution-Local%20CPU%20%7C%20Modal%20A10G%20GPU-orange.svg)](macro_forecast_engine/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Zero-Trust Audit PASSED](https://img.shields.io/badge/Audit-PASSED-brightgreen.svg)](FINRIX_SECURITY_AND_QA_AUDIT_REPORT.md)
[![14-Strategy Engine](https://img.shields.io/badge/Strategies-14-brightgreen.svg)](macro_forecast_engine/)





https://github.com/user-attachments/assets/b600ad35-9752-46c7-a912-12deb9714708






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

Real-time dollar risk math with standardized Gold pip sizing, pip-based
stop-loss/take-profit calculation, equity compounding, transaction-cost-aware
trade sizing, and fractional-Kelly allocation across 14 strategies.

### Genuine Trade-Based Backtesting

The timemachine engine simulates signal entry, open-position tracking,
stop-loss/take-profit exits, signal reversal, 1.5-pip spread, and slippage.
Metrics use closed trades and the dollar equity curve rather than static
per-bar percentage returns.

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

## Quality Assurance

The repository includes a real pytest suite covering dynamic risk sizing,
lookahead-safe trade lifecycle execution, all 14 indicator strategies,
timeframe-specific caching, non-finite JSON coercion, and configuration
sanitization.

```text
207 passed
```

Run the checks with:

```bash
python -m pytest -q
python -m compileall -q app.py macro_forecast_engine
```

Finrix Screenshots:

<img width="283" height="304" alt="Screenshot 2026-11-08 105821" src="https://github.com/user-attachments/assets/919fb0d2-2d04-4172-a401-9f4caab112c7" />
<img width="282" height="281" alt="Screenshot 2026-11-08 105740" src="https://github.com/user-attachments/assets/d5558b17-58df-434c-9fda-0e8376d08090" />
<img width="275" height="259" alt="Screenshot 2026-11-08 105707" src="https://github.com/user-attachments/assets/c11f30cd-bc4c-45fa-9ffa-6ab9fde55a88" />
<img width="276" height="128" alt="Screenshot 2026-11-07 225107" src="https://github.com/user-attachments/assets/b0487285-9d31-4dd7-8191-e7b53a166eeb" />
<img width="285" height="529" alt="Screenshot 2026-11-07 225031" src="https://github.com/user-attachments/assets/894637f0-90d2-4c23-9465-83352218779e" />




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
.github/
├── workflows/
│   └── ci.yml            # GitHub Actions CI pipeline
└── ISSUE_TEMPLATE/
    ├── bug_report.md     # Bug report template
    └── feature_request.md # Feature request template
.github/PULL_REQUEST_TEMPLATE.md   # PR checklist template
CONTRIBUTING.md            # Open-source guidelines
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
