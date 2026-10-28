# Security Review

## Scope

Reviewed the source code, configuration, dependency declarations, and tracked
runtime artifacts in `carnoba/trading-prediction`.

## Findings

| # | Severity | File | Lines | Vulnerability | Confidence |
|---|----------|------|-------|---------------|------------|
| 1 | HIGH | `mt5-agent/dashboard.py` | 40-54, 212-214, 339-347 | The Streamlit dashboard exposes live account data and order execution without application authentication or authorization. | 9/10 |
| 2 | MEDIUM | `mt5-agent/macro_forecast_engine/modal_engine.py` | 424-447 | `FRED_API_KEY` is passed as a Modal function argument and may be exposed through invocation metadata or logs. | 8/10 |
| 3 | MEDIUM | `mt5-agent/macro_forecast_engine/requirements.txt`, `mt5-agent/macro_forecast_engine/modal_engine.py`, `mt5-agent/smc_modal_backtest.py` | 1-8; 35-45; 48-51 | Runtime and Modal image dependencies are unpinned, increasing supply-chain and reproducibility risk. | 8/10 |

## Credential scan

No hardcoded API keys, broker passwords, or private tokens were found in the
scanned source and configuration files. `.env.example` contains placeholders
only. Credentials should continue to be loaded from the runtime environment or
secret manager.

## Recommendations

1. Require strong authentication and server-side authorization before
   connecting to MT5 or exposing order controls. Bind the dashboard to a
   private interface or VPN and disable public tunnels.
2. Store `FRED_API_KEY` in a Modal Secret and read it inside the worker rather
   than passing it as an invocation argument.
3. Pin exact dependency versions, preferably with hashes and a lockfile, and
   review dependency updates before deployment.
