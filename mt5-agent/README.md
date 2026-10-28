# MT5 Data + Backtest Pipeline (Step 1 — zero execution risk)

Read-only research stack: pulls history from your MT5 terminal and backtests
two baseline strategies with realistic costs. **Never places orders.**

## Prerequisites

1. Python 3.13 64-bit (already installed) + packages:
   ```
   pip install MetaTrader5 backtesting pandas
   ```
   (already present on this machine)
2. **MetaTrader 5 terminal installed + logged in manually** (demo account fine).
   Current status on this PC: **NOT FOUND** — install from your broker
   (Exness / OctaFX / IC Markets etc.), log in once, then run `env_check.py`.
3. Fill `BROKER_SERVER` in `config.py` with your server name
   (shown in MT5: File -> Login to Trade Account).

## Files

| File | Purpose |
|---|---|
| `config.py` | symbols, timeframes, costs, split settings |
| `env_check.py` | verifies terminal/account/data availability (read-only) |
| `data_pull.py` | pulls OHLCV bars (M15/H1/H4 since 2020) + raw ticks -> `data/*.csv` |
| `mt5_data.py` | loader with fallback to bundled sample data |
| `strategies.py` | `EMACrossATR` and `OrderBlockRetest` baselines |
| `run_backtest.py` | runs IS/OOS/FULL backtests + metrics + verdict |

## Run order

```
export DASHBOARD_PASSWORD='use-a-long-random-password'
python env_check.py                 # 1. verify MT5 connection
python data_pull.py                 # 2. pull history -> data/*.csv
python run_backtest.py              # 3. both strategies on EURUSD H1
python run_backtest.py --strategy smc --symbol XAUUSD --tf M15
python run_backtest.py --sample     # no MT5 needed (bundled sample data)
```

The Streamlit dashboard refuses access unless `DASHBOARD_PASSWORD` is set and
the password is entered successfully. Keep the dashboard bound to localhost
or a private network; it is not intended for public exposure.

## Realism settings (config.py)

- `SPREAD = 0.0002` — relative bid-ask spread on every fill (buy at ask, sell at bid)
- `COMMISSION = 0.0005` — round-trip commission rate
- 1% equity risk per trade; **hard SL on every position**; fills on next bar open
  (no same-bar / look-ahead fills)

## Report columns

- **win%** – winning trades / total
- **PF** – profit factor (gross win / gross loss)
- **expect$** – average cash PnL per trade (expectancy)
- **return%**, **MaxDD%**, **Sharpe** – from backtesting.py stats
- **IS** = first 70% of data (in-sample), **OOS** = last 30% (**unseen** validation)

A strategy only "passes" Step 1 if **OOS PF >= 1.1 with >= 30 trades**.
OOS results ≈ IS results is the honest signal; a big drop = overfitting.

## Next gates (agreed framework)

1. ~~Backtest~~ ← you are here
2. Port the validated logic to **MQL5 + MT5 Strategy Tester** (tick-level)
3. **>= 2-4 weeks demo forward-test** of the exact execution code
4. Live only after 2-3 pass, with hard guardrails (lot cap, daily-loss halt)

## Known limitations

- Constant spread (no variable spread/news spikes); MT5 tick backtest covers that later
- No swap/overnight financing modeled
- Simplified SMC baseline (single-liquidity model) — educational, not production SMC
- `backtesting.py` equity assumes instantaneous fills at next open; slippage beyond
  spread is not modeled (addressed by MT5 tick-level tester in Step 2)
