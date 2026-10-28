## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                              │
├─────────────────────┬──────────────────────┬───────────────────────┤
│                     │                    │                       │
│   FRED API          │   ForexFactory       │   yfinance /           │
│   (TIPS Yields)     │   (High-Impact News) │   MT5 Broker Data      │
│                     │                    │                       │
├─────────────────────┼──────────────────────┼───────────────────────┤
│                     │                    │                       │
│   FinBERT Model     │   News Scraper       │   VectorBT Engine      │
│   (Sentiment)       │                       │   (Backtesting)        │
│                     │                    │                       │
├─────────────────────┼──────────────────────┼───────────────────────┤
│                     │                    │                       │
│   NeuralForecast    │   Macro Indicators   │   Execution Engine     │
│   (TFT Forecast)    │   (RSI, ADX, etc.)   │   (Order Management)   │
│                     │                    │                       │
└─────────────────────┴──────────────────────┴───────────────────────┘
# Institutional Macro-Gated Quantitative XAUUSD Forecasting & Execution Engine

![Project Banner](https://img.shields.io/badge/Status-In%20Development-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Project Overview

The **Institutional Macro-Gated Quantitative XAUUSD Forecasting & Execution Engine** is a comprehensive quantitative trading system that integrates multiple data sources and machine learning models to generate high-probability XAUUSD (Gold) trading signals.

### Core Features

- **NeuralForecast TFT**: Temporal Fusion Transformer for multi-horizon time series forecasting
- **FinBERT Sentiment**: Real-time sentiment analysis from financial news
- **FRED TIPS Yields**: U.S. Treasury Inflation-Protected Securities yield curve data
- **M15 RSI 21 Execution**: Mean-reversion execution on 15-minute timeframe
- **Modal GPU Deployment**: Scalable cloud GPU inference
- **Out-of-Sample Robustness**: 70/30 chronological train/test split with stability scoring
- **Risk Controls**: Daily drawdown kill-switch, position sizing, leverage caps

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                              │
├─────────────────────┬──────────────────────┬───────────────────────┤
│                     │                    │                       │
│   FRED API          │   ForexFactory       │   yfinance /           │
│   (TIPS Yields)     │   (High-Impact News) │   MT5 Broker Data      │
│                     │                    │                       │
├─────────────────────┼──────────────────────┼───────────────────────┤
│                     │                    │                       │
│   FinBERT Model     │   News Scraper       │   VectorBT Engine      │
│   (Sentiment)       │                       │   (Backtesting)        │
│                     │                    │                       │
├─────────────────────┼──────────────────────┼───────────────────────┤
│                     │                    │                       │
│   NeuralForecast    │   Macro Indicators   │   Execution Engine     │
│   (TFT Forecast)    │   (RSI, ADX, etc.)   │   (Order Management)   │
│                     │                    │                       │
└─────────────────────┴──────────────────────┴───────────────────────┘

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- Git
- MetaTrader 5 terminal (for live trading features)
- ForexFactory account (for news scraping - public, no account needed)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/macro-quant-engine.git
cd macro-quant-engine
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

**Required `.env` variables:**

```
FRED_API_KEY=your_fred_api_key_here      # From https://fred.stlouisfed.org/docs/api/fred/auth.html
MODAL_TOKEN=your_modal_token_here        # From https://modal.com/docs/reference/token
```

**Optional `.env` variables:**

```
BROKER_SERVER=MetaQuotes-Demo           # MT5 broker server name
MT5_LOGIN=your_login                    # MT5 terminal login
MT5_PASSWORD=your_password              # MT5 terminal password
FORECAST_HORIZON=monthly                # "monthly" or "yearly"
TFT_HIDDEN_SIZE=64                      # NeuralForecast TFT parameter
TFT_LAGS=24                             # NeuralForecast TFT parameter
FINBERT_MODEL=ProsusAI/finbert          # Sentiment analysis model
DEFAULT_TIMEFRAME=M15                   # Trading timeframe
MAX_DRAWDOWN_PCT=2.0                    # Daily max drawdown %
MAX_POSITION_SIZE=0.1                   # Max position size fraction
MODE=monthly                            # "monthly" or "yearly"
```

### 5. Verify Installation

```bash
python -m pytest tests/ -v
# Or run a quick test:
python import_test.py
python test_import.py
python test_run.py
```

## 🙏 Acknowledgments

- **NeuralForecast**: Time series forecasting library
- **pandas_ta**: Technical analysis indicators
- **vectorbt**: Portfolio backtesting engine
- **FinBERT**: Financial sentiment analysis
- **FRED**: Federal Reserve Economic Data
- **MetaTrader 5**: Broker platform
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Quick import verification
python import_test.py
python test_import.py  
python test_run.py

# Linting
pylint mt5-agent/ macro_forecast_engine/

# Syntax check
python -m py_compile mt5-agent/*.py macro_forecast_engine/*.py
```
## 🔒 Security & Best Practices

### Environment Variable Management

- **Never commit `.env` files** - They contain sensitive API keys
- Use `.env.example` to document required variables without exposing secrets
- Load variables via `os.getenv()` or `python-dotenv`
- Rotate API keys regularly

### Data Sensitivity

- MT5 login credentials are read at runtime from environment
- Broker server names are non-sensitive but configurable
- All data paths are relative to project root

### Hardened Defaults

- Dashboard runs in READ-ONLY mode by default
- Daily drawdown kill-switch at 2.0% equity
- Mandatory Stop-Loss and Take-Profit on every order
- No hardcoded broker credentials in source
```
## 🎮 CLI Commands

### Main Engine CLI

```bash
python -m mt5_agent.gold_backtest_engine --help
```

### Supported Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--period` | Data period (e.g., "60d", "90d", "1y") | "730d" |
| `--interval` | Timeframe: "1h" or "1d" | "1h" |
| `--output` | Output JSON file path | `mt5-agent/data/top_gold_strategies.json` |
| `--long-only` | Disable short selling | `False` |
| `--mode` | Forecast horizon mode | "monthly" |

### Mode Examples

```bash
# Monthly forecast mode (default)
python -m mt5_agent.gold_backtest_engine --mode monthly --period "60d" --interval "1h"

# Yearly forecast mode
python -m mt5_agent.gold_backtest_engine --mode yearly --period "365d" --interval "1d"

# Long-only (no short selling)
python -m mt5_agent.gold_backtest_engine --mode monthly --long-only

# Custom output path
python -m mt5_agent.gold_backtest_engine --mode monthly --output ./data/my_strategies.json
```

### Modal GPU CLI

```bash
# Run forecast on Modal GPU
modal run forecast_engine.py --mode yearly

# Run with custom horizon
modal run forecast_engine.py --mode monthly --gpu A100
```
## 🚀 Modal GPU Setup Guide

### Prerequisites

- Modal account and token
- GPU-enabled container access

### 1. Set Modal Token

```bash
export MODAL_TOKEN="your_modal_token_here"
# Or add to .env file: MODAL_TOKEN=your_modal_token_here
```

### 2. Deploy to Modal

```bash
modal deploy modal_config.py
# Or run inline:
modal run forecast_engine.py --mode monthly
```

### 3. Modal Configuration Example

```python
# modal_config.py
from macro_forecast_engine.config import BASE_DIR, FORECAST_HORIZON

horizon = FORECAST_HORIZON  # "monthly" or "yearly"

stub = modal.Stub("macro-forecast")

@stub.function(gpu="A100", image=modal.Image.debian_python().pip_install(
    "numpy", "pandas", "pytorch", "neuralforecast", "finbert-emoji"
))
def run_forecast():
    """Run TFT forecast on GPU."""
    # ... forecast logic
    pass

if __name__ == "__main__":
    modal.cli.run()
```

### 4. Access Modal Results

```bash
modal result show <run-id>
# Or fetch via API
```