"""FINRIX interactive Rich terminal for the XAUUSD macro forecast engine."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    import questionary
except ImportError:  # pragma: no cover
    questionary = None
try:
    from InquirerPy import inquirer
except ImportError:  # pragma: no cover
    inquirer = None

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from macro_forecast_engine.data_loader import fetch_live_news
from macro_forecast_engine.data_processor import analyze_news_sentiment
from macro_forecast_engine.finrix_config import load_config, save_config
from macro_forecast_engine.indicator_params import (
    DEFAULT_INDICATOR_PARAMS, load_indicator_params, save_indicator_params,
)
from macro_forecast_engine.trader_execution import build_execution_plan
from macro_forecast_engine.trader_risk import size_position
from macro_forecast_engine.json_utils import _json_safe
from macro_forecast_engine.trader_timemachine import (
    STRATEGIES, TIMEFRAME_FETCH, normalize_timeframe_prices, run_strategy_matrix,
    replay_forecasts, strategy_confidence,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CONFIG_PATH = DATA_DIR / "finrix_config.json"
ARTIFACT_FILENAMES = {
    "daily": "macro_daily_forecast.json",
    "monthly": "macro_month_forecast.json",
    "yearly": "macro_year_forecast.json",
}
CONFIG = load_config(CONFIG_PATH)
INDICATOR_PARAMS = load_indicator_params()
console = Console()


@contextmanager
def _status(message: str, spinner: str = "dots"):
    """Show one status line, updating in place on a real terminal.

    Rich's live display relies on ANSI cursor movement. In IDE consoles and
    redirected output it can redraw the same text on a new line repeatedly, so
    fall back to a single static line whenever stdout is not a TTY.
    """
    text = f"[green]{message}[/green]"
    if console.is_terminal:
        with console.status(text, spinner=spinner, refresh_per_second=4):
            yield
    else:
        console.print(text)
        yield


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        value = value.to_dict("records")
    if isinstance(value, dict):
        if isinstance(value.get("records"), list):
            value = value["records"]
        elif value and all(isinstance(item, dict) for item in value.values()):
            value = list(value.values())
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, dict)]
    return []


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _fmt_money(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"${number:,.4f}" if math.isfinite(number) else "n/a"


def _normalize_forecast_payload(payload: Any) -> dict[str, Any]:
    """Coerce Modal/local artifacts into the schema the UI renders.

    Older artifacts and partial Modal responses may expose ``forecast_df``,
    ``volatility_corridors``, ``signals``, or ``diagnostics`` instead of (or
    alongside) the current keys. Normalizing here keeps every renderer free of
    KeyError/TypeError crashes.
    """
    if not isinstance(payload, dict):
        raise ValueError("Forecast artifact must be a JSON object")
    normalized = dict(payload)

    forecast = _as_records(normalized.get("forecast"))
    if not forecast:
        forecast = _as_records(normalized.get("forecast_df"))
    signals = normalized.get("signals")
    if isinstance(signals, list):
        for row, signal in zip(forecast, signals):
            row.setdefault("signal_bias", str(signal))
    elif isinstance(signals, dict):
        for row in forecast:
            if "signal_bias" not in row and row.get("date") in signals:
                row["signal_bias"] = str(signals[row["date"]])
    for row in forecast:
        row.setdefault("date", "n/a")
        row.setdefault("target_q50", 0.0)
        row.setdefault("expected_low_q10", 0.0)
        row.setdefault("expected_high_q90", 0.0)
        row.setdefault("signal_bias", "neutral")
    normalized["forecast"] = forecast

    monte_carlo = _as_mapping(normalized.get("monte_carlo"))
    corridors = normalized.get("volatility_corridors")
    if isinstance(corridors, list) and corridors and isinstance(corridors[0], dict):
        corridors = corridors[0]
    corridors = _as_mapping(corridors)
    for key in ("target_price", "target_low", "target_high"):
        if key not in monte_carlo and key in corridors:
            monte_carlo[key] = corridors[key]
    if forecast:
        monte_carlo.setdefault("target_price", forecast[0].get("target_q50"))
        monte_carlo.setdefault("target_low", forecast[0].get("expected_low_q10"))
        monte_carlo.setdefault("target_high", forecast[0].get("expected_high_q90"))
    normalized["monte_carlo"] = monte_carlo

    diagnostics = normalized.get("diagnostics", normalized.get("training"))
    normalized["diagnostics"] = _as_mapping(diagnostics)
    normalized["training"] = _as_mapping(normalized.get("training"))

    for key in ("data", "risk_levels", "position_sizing", "smc",
                "risk_configuration", "ingestion"):
        normalized[key] = _as_mapping(normalized.get(key))
    normalized.setdefault("mode", "unknown")
    if not normalized.get("directional_bias"):
        normalized["directional_bias"] = str(monte_carlo.get("directional_bias", "neutral"))
    if normalized.get("confidence_score") is None:
        normalized["confidence_score"] = monte_carlo.get("confidence_score", 0.0)
    if not normalized.get("buy_zone"):
        normalized["buy_zone"] = normalized["smc"].get("buy_zone", [])
    if not normalized.get("sell_zone"):
        normalized["sell_zone"] = normalized["smc"].get("sell_zone", [])
    return normalized


def refresh_config() -> dict[str, Any]:
    global CONFIG
    CONFIG = load_config(CONFIG_PATH)
    return CONFIG


def refresh_indicator_params() -> dict[str, dict[str, float | int]]:
    global INDICATOR_PARAMS
    INDICATOR_PARAMS = load_indicator_params()
    return INDICATOR_PARAMS


def customize_indicator_params() -> None:
    global INDICATOR_PARAMS
    params = refresh_indicator_params()
    choices = [
        "[1] Use Standard MT5/TradingView Parameters (Defaults)",
        "[2] Customize Indicator Parameters",
    ]
    if inquirer is not None:
        choice = inquirer.select(message="Indicator parameters", choices=choices).execute()
    else:
        choice = choose("Indicator parameters", choices)
    if choice.startswith("[1]"):
        INDICATOR_PARAMS = save_indicator_params(DEFAULT_INDICATOR_PARAMS)
        console.print("[green]Standard indicator parameters applied.[/green]")
        return
    fields = (
        ("rsi", "period", 2, 100, "RSI period"),
        ("rsi", "overbought", 50, 99, "RSI overbought"),
        ("rsi", "oversold", 1, 50, "RSI oversold"),
        ("ema", "fast_period", 2, 200, "EMA fast period"),
        ("ema", "slow_period", 3, 400, "EMA slow period"),
        ("ema", "trend_period", 10, 500, "EMA trend period"),
        ("macd", "fast", 2, 100, "MACD fast"),
        ("macd", "slow", 3, 200, "MACD slow"),
        ("macd", "signal", 2, 100, "MACD signal"),
        ("bollinger", "period", 2, 200, "Bollinger period"),
        ("bollinger", "std_dev", 0.5, 6.0, "Bollinger standard deviation"),
        ("supertrend", "atr_period", 2, 100, "Supertrend ATR period"),
        ("supertrend", "factor", 0.5, 10.0, "Supertrend factor"),
        ("stochastic", "k_period", 2, 100, "Stochastic K period"),
        ("stochastic", "d_period", 2, 50, "Stochastic D period"),
        ("stochastic", "slowing", 1, 50, "Stochastic slowing"),
        ("adx", "period", 2, 100, "ADX period"),
        ("adx", "threshold", 1, 100, "ADX threshold"),
        ("atr", "period", 2, 100, "ATR period"),
        ("ichimoku", "tenkan", 2, 100, "Ichimoku Tenkan"),
        ("ichimoku", "kijun", 2, 200, "Ichimoku Kijun"),
        ("ichimoku", "senkou_b", 2, 300, "Ichimoku Senkou B"),
        ("smc", "fvg_min_pip_gap", 0.1, 20.0, "SMC FVG minimum pip gap"),
        ("smc", "order_block_lookback", 2, 200, "SMC order block lookback"),
    )
    for group, key, low, high, label in fields:
        current = str(params[group][key])
        value = ask(f"{label} ({low}-{high})", current)
        try:
            numeric = float(value)
            if not low <= numeric <= high:
                raise ValueError
            params[group][key] = int(numeric) if numeric.is_integer() else numeric
        except ValueError:
            console.print(f"[yellow]Keeping {label}={current}; invalid value.[/yellow]")
    INDICATOR_PARAMS = save_indicator_params(params)
    console.print("[green]Indicator overrides saved to data/finrix_indicator_params.json.[/green]")


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def ask(prompt: str, default: str = "") -> str:
    if questionary is not None:
        return questionary.text(prompt, default=default).ask() or default
    return input(f"{prompt} [{default}]: ").strip() or default


def choose(message: str, choices: list[str]) -> str:
    if questionary is not None:
        return questionary.select(message, choices=choices).ask() or choices[-1]
    console.print("\n".join(f"{i}. {value}" for i, value in enumerate(choices, 1)))
    value = ask(message, str(len(choices)))
    try:
        return choices[int(value) - 1]
    except (ValueError, IndexError):
        return choices[-1]


def choose_many(message: str, choices: list[str]) -> list[str]:
    if questionary is not None:
        selected = questionary.checkbox(message, choices=choices).ask()
        selected = selected or choices[:1]
        normalized: list[str] = []
        for item in selected:
            value = getattr(item, "value", item)
            title = getattr(item, "title", value)
            candidate = value if value in choices else title
            if candidate in choices and candidate not in normalized:
                normalized.append(candidate)
        return normalized or choices[:1]
    console.print("\n".join(f"{i}. {value}" for i, value in enumerate(choices, 1)))
    raw = ask(f"{message} (comma-separated numbers)", "1")
    selected: list[str] = []
    for token in raw.split(","):
        try:
            value = choices[int(token.strip()) - 1]
        except (ValueError, IndexError):
            continue
        if value not in selected:
            selected.append(value)
    return selected or choices[:1]


def banner() -> None:
    refresh_config()
    clear_screen()
    art = Text(
        "+--------------------------------------------------------------+\n"
        "|          FINRIX QUANT TRADING TERMINAL (XAUUSD)            |\n"
        "+--------------------------------------------------------------+",
        style="bold green",
    )
    console.print(Panel(art, border_style="green", padding=(1, 2)))
    status = Table.grid(padding=(0, 2))
    status.add_column(style="green")
    status.add_column()
    status.add_row("SYSTEM", "[bold green]ONLINE[/bold green]")
    environment = CONFIG["execution_environment"]
    status.add_row("EXECUTION", "[bold green]MODAL GPU (A10G)[/bold green]"
                   if environment == "modal" else "[bold green]LOCAL CPU[/bold green]")
    status.add_row("DEFAULT EQUITY", f"${CONFIG['account_equity']:,.2f}")
    status.add_row("RISK / TRADE", f"{CONFIG['risk_per_trade_pct']:.2f}%")
    status.add_row("ACTIVE SKILLS", "TFT | SMC | FinBERT | Kelly | Volatility Parity | Timemachine")
    status.add_row("DATA ROOT", str(DATA_DIR))
    console.print(Panel(status, title="Terminal Status", border_style="green"))


def artifact_path(mode: str) -> Path:
    return DATA_DIR / ARTIFACT_FILENAMES.get(mode, ARTIFACT_FILENAMES["monthly"])


def run_forecast(mode: str, force: bool = True) -> dict[str, Any]:
    refresh_config()
    if mode not in ARTIFACT_FILENAMES:
        raise ValueError(f"Unknown forecast mode: {mode}")
    path = artifact_path(mode)
    if not force and path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            age_seconds = pd.Timestamp.now().timestamp() - path.stat().st_mtime
            expected_environment = (
                "Modal GPU A10G" if CONFIG["execution_environment"] == "modal"
                else "Local CPU"
            )
            if age_seconds < 900 and cached.get("execution_environment") == expected_environment:
                return _normalize_forecast_payload(cached)
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    if CONFIG["execution_environment"] == "modal":
        command = [sys.executable, "-m", "modal", "run",
                   "macro_forecast_engine/modal_engine.py", "--mode", mode,
                   "--account-equity", str(CONFIG["account_equity"]),
                   "--risk-per-trade-pct", str(CONFIG["risk_per_trade_pct"])]
        status_text = f"Running {mode} forecast on Modal A10G Cloud GPU..."
        error_title = "Modal Forecast Error"
    else:
        command = [sys.executable, "-m", "macro_forecast_engine.run_engine",
                   "--mode", mode, "--environment", "local",
                   "--account-equity", str(CONFIG["account_equity"]),
                   "--risk-per-trade-pct", str(CONFIG["risk_per_trade_pct"])]
        status_text = f"Running {mode} forecast on Local CPU (Python Engine)..."
        error_title = "Local Forecast Error"
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with _status(status_text):
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                   encoding="utf-8", errors="replace", env=environment)
    if completed.returncode != 0:
        console.print(Panel("[ERR] " + completed.stdout + completed.stderr,
                            title=error_title, border_style="red", box=box.ASCII))
        raise RuntimeError(
            f"{CONFIG['execution_environment']} forecast failed with exit code "
            f"{completed.returncode}"
        )
    if not path.exists():
        raise FileNotFoundError(f"Forecast artifact was not created: {path}")
    try:
        artifact = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Forecast artifact could not be parsed: {path} ({error})"
        ) from error
    return _normalize_forecast_payload(artifact)


def print_execution_card(payload: dict[str, Any], equity: float) -> None:
    risk = _as_mapping(payload.get("risk_levels"))
    sizing = _as_mapping(payload.get("position_sizing"))
    bias = str(payload.get("directional_bias", "neutral")).lower()
    action = "BUY" if bias == "bullish" else "SELL" if bias == "bearish" else "NO TRADE (Wait for Zone)"
    table = Table(title="Today's Best Trade - Instant Execution Card", box=box.ASCII)
    table.add_column("Metric", style="green")
    table.add_column("Value", style="bold white")
    table.add_row("Recommended Action", action)
    table.add_row("Ideal Entry Zone", str(risk.get("entry_zone", "n/a")))
    table.add_row("Dynamic SL", _fmt_money(risk.get("stop_loss")))
    table.add_row("TP1 / TP2 / TP3", " / ".join(_fmt_money(risk.get(key)) for key in
                                                ("take_profit_1", "take_profit_2", "take_profit_3")))
    table.add_row("Lot Size", f"{_to_float(sizing.get('lot_size')):.2f} Lots "
                              f"({_to_float(sizing.get('ounces')):.1f} oz)")
    table.add_row("Actual Risk", f"{_to_float(sizing.get('actual_risk_pct')):.4f}%")
    if sizing.get("risk_warning"):
        table.add_row("Risk Alert", str(sizing["risk_warning"]))
    table.add_row("Confidence", f"{_to_float(payload.get('confidence_score')):.2f}%")
    reward_ratio = _to_float(risk.get("risk_reward", sizing.get("reward_risk", 0.0)))
    table.add_row("Risk : Reward", f"1 : {reward_ratio:.2f}")
    console.print(table)


def show_forecast(payload: dict[str, Any]) -> None:
    refresh_config()
    payload = _normalize_forecast_payload(payload)
    data = payload["data"]
    monte_carlo = payload["monte_carlo"]
    risk = payload["risk_levels"]
    environment = "MODAL GPU" if CONFIG["execution_environment"] == "modal" else "LOCAL CPU"
    console.print(Panel(
        f"[bold green]{str(payload.get('mode', 'unknown')).capitalize()}[/bold green] | "
        f"{data.get('horizon', 'n/a')} business days | "
        f"Engine: {environment} | Equity: ${_to_float(CONFIG.get('account_equity')):,.2f} | "
        f"Risk: {_to_float(CONFIG.get('risk_per_trade_pct')):.2f}%\n"
        f"Bias: {str(payload.get('directional_bias', 'neutral')).upper()} "
        f"({_to_float(payload.get('confidence_score')):.2f}%)\n"
        f"Target: {_fmt_money(monte_carlo.get('target_price'))}  "
        f"Corridor: {_fmt_money(monte_carlo.get('target_low'))} - "
        f"{_fmt_money(monte_carlo.get('target_high'))}\n"
        f"SMC Buy: {payload.get('buy_zone', 'n/a')}  SMC Sell: {payload.get('sell_zone', 'n/a')}\n"
        f"SL: {_fmt_money(risk.get('stop_loss'))}  "
        f"TP1/TP2/TP3: {_fmt_money(risk.get('take_profit_1'))} / "
        f"{_fmt_money(risk.get('take_profit_2'))} / "
        f"{_fmt_money(risk.get('take_profit_3'))}",
        title="Forecast Overview",
        border_style="green", box=box.ASCII,
    ))
    table = Table(title="TFT Forward Path", box=box.ASCII)
    for column in ("Date", "q10", "q50", "q90", "Signal"):
        table.add_column(column, justify="right" if column != "Signal" else "left")
    for row in payload["forecast"]:
        table.add_row(str(row.get("date", "n/a")),
                      _fmt_money(row.get("expected_low_q10")),
                      _fmt_money(row.get("target_q50")),
                      _fmt_money(row.get("expected_high_q90")),
                      str(row.get("signal_bias", "neutral")).upper())
    console.print(table)


def today_signal() -> None:
    refresh_config()
    payload = _normalize_forecast_payload(run_forecast("daily", force=False))
    equity = _to_float(CONFIG.get("account_equity"))
    forecast = payload["forecast"]
    if not forecast:
        raise ValueError("Forecast artifact does not contain any forecast rows")
    bias = str(payload.get("directional_bias", "neutral")).lower()
    risk = payload["risk_levels"]
    entry_zone = risk.get("entry_zone")
    if not isinstance(entry_zone, (list, tuple)) or len(entry_zone) < 2:
        entry_zone = [forecast[0].get("expected_low_q10"), forecast[0].get("expected_high_q90")]
    entry = _to_float(entry_zone[0] if bias == "bullish" else entry_zone[1])
    payload["risk_levels"] = build_execution_plan(
        entry, forecast, payload["smc"], bias,
    )
    row = forecast[0]
    sizing = size_position(
        entry,
        _to_float(row.get("expected_low_q10")),
        _to_float(row.get("target_q50")),
        _to_float(row.get("expected_high_q90")),
        equity,
        risk_fraction=max(0.0, _to_float(CONFIG.get("risk_per_trade_pct"), 1.0)) / 100.0,
        kelly_fraction=0.25,
        micro_lot_floor=_to_float(CONFIG.get("micro_lot_floor"), 0.01),
        stop_loss=_to_float(payload["risk_levels"].get("stop_loss")),
    )
    payload["position_sizing"] = sizing
    print_execution_card(payload, equity)


def news_report() -> None:
    with _status("Fetching macro news and FinBERT sentiment..."):
        news = fetch_live_news()
        sentiment_data = _as_mapping(analyze_news_sentiment(news))
        sentiment = _to_float(sentiment_data.get("score"), float("nan"))
        velocity = _to_float(sentiment_data.get("velocity"), float("nan"))
    table = Table(title="Macro News & FinBERT", box=box.ROUNDED)
    table.add_column("Metric", style="green")
    table.add_column("Value")
    table.add_row("Headlines", str(len(news)))
    table.add_row("Hawkish / Dovish Score", f"{sentiment:+.4f}" if not pd.isna(sentiment) else "unavailable")
    table.add_row("Bias", "BULLISH" if sentiment > 0.05 else "BEARISH"
                  if sentiment < -0.05 else "NEUTRAL")
    table.add_row("Sentiment Velocity", f"{velocity:+.4f}"
                  if not pd.isna(velocity) else "unavailable")
    console.print(table)
    if not news.empty:
        for title in news["headline"].head(8).tolist():
            console.print(f"  - {title}")
    else:
        console.print("  No gold macro headlines were returned by the approved feeds.")


def execution_lookup() -> None:
    refresh_config()
    payload = _normalize_forecast_payload(run_forecast("daily"))
    forecast = payload["forecast"]
    if not forecast:
        raise ValueError("Forecast artifact does not contain any forecast rows")
    row = forecast[0]
    default_price = _to_float(row.get("target_q50"))
    try:
        price = float(ask("Current market price ($)", str(default_price)))
    except (TypeError, ValueError) as error:
        raise ValueError("Current market price must be numeric") from error
    equity = _to_float(CONFIG.get("account_equity"))
    risk_percent = _to_float(CONFIG.get("risk_per_trade_pct"), 1.0)
    plan = build_execution_plan(price, forecast, payload["smc"],
                                str(payload.get("directional_bias", "neutral")))
    sizing = size_position(price, _to_float(row.get("expected_low_q10")),
                           _to_float(row.get("target_q50")),
                           _to_float(row.get("expected_high_q90")), equity,
                           risk_fraction=max(0.0, risk_percent) / 100.0,
                           kelly_fraction=0.25,
                           micro_lot_floor=_to_float(CONFIG.get("micro_lot_floor"), 0.01),
                           stop_loss=_to_float(plan.get("stop_loss")))
    sizing["pip_risk"] = round(abs(price - _to_float(plan.get("stop_loss"))) / 0.10, 2)
    payload["risk_levels"] = plan
    payload["position_sizing"] = sizing
    print_execution_card(payload, equity)


def backtester() -> None:
    refresh_config()
    customize_indicator_params()
    strategies = choose_many("Select strategies to test", list(STRATEGIES))
    timeframes = choose_many("Select timeframes to test", ["15m", "1h", "4h", "1D"])
    console.print(
        f"[green]Selected strategies:[/green] {', '.join(strategies)}\n"
        f"[green]Selected timeframes:[/green] {', '.join(timeframes)}"
    )
    from macro_forecast_engine.data_loader import fetch_market_data
    with _status("Loading historical XAUUSD data..."):
        frames = []
        for timeframe in timeframes:
            period, interval = TIMEFRAME_FETCH[timeframe]
            market = fetch_market_data(period, interval=interval)
            market_prices = market[[column for column in (
                "gold_open", "gold_high", "gold_low", "gold_close"
            ) if column in market]]
            frames.append((timeframe, normalize_timeframe_prices(market_prices, timeframe)))
    with _status("Replaying strategies across the selected timeframes..."):
        result = pd.concat(
            [run_strategy_matrix(
                prices, strategies, [timeframe], INDICATOR_PARAMS,
                CONFIG["account_equity"], CONFIG["risk_per_trade_pct"],
            ) for timeframe, prices in frames],
            ignore_index=True,
        )
    table = Table(title="Lookahead-Safe Multi-Strategy / Multi-Timeframe Replay", box=box.ASCII)
    for column in ("Strategy", "Timeframe", "Win Rate %", "Profit Factor",
                   "Max Drawdown %", "Sharpe", "Net Expected Return %"):
        table.add_column(column, style="green" if column in ("Strategy", "Timeframe") else None,
                         justify="right" if column not in ("Strategy", "Timeframe") else "left")
    for row in result.to_dict("records"):
        profit_factor = "inf" if math.isinf(row["profit_factor"]) else f"{row['profit_factor']:.2f}"
        table.add_row(row["strategy"], row["timeframe"], f"{row['win_rate']:.2f}",
                      profit_factor, f"{row['max_drawdown']:.2f}",
                      f"{row['sharpe_ratio']:.2f}", f"{row['net_expected_return']:.2f}")
    console.print(table)
    console.print(
        f"[green]Account equity:[/green] ${CONFIG['account_equity']:,.2f} | "
        f"[green]Risk per trade:[/green] {CONFIG['risk_per_trade_pct']:.2f}%"
    )


def _historical_log_return_forecast(history: pd.Series, horizon: int) -> list[float]:
    """Use only the expanding history to produce a backend trend forecast."""
    returns = pd.Series(history).pct_change().dropna().tail(252)
    drift = float((returns + 1.0).clip(lower=1e-9).map(math.log).mean())
    return (float(history.iloc[-1]) * pd.Series(
        np.exp(np.cumsum([drift] * horizon))
    )).tolist()


def ensemble() -> None:
    refresh_config()
    customize_indicator_params()
    path = DATA_DIR / "modal_ensemble.json"
    if CONFIG["execution_environment"] == "modal":
        proceed = ask(
            "[WARNING] Parallel multi-strategy backtesting on Modal A10G consumes cloud credits. "
            "Proceed? (Y/N)", "N",
        ).strip().lower()
        if proceed not in {"y", "yes"}:
            console.print("[yellow]Cancelled; no Modal jobs were submitted.[/yellow]")
            return
        with _status("Running parallel strategy ensemble on Modal A10G..."):
            command = [sys.executable, "-m", "modal", "run",
                       "macro_forecast_engine/modal_engine.py", "--mode", "ensemble",
                       "--market-period", "5y",
                       "--account-equity", str(CONFIG["account_equity"]),
                       "--risk-per-trade-pct", str(CONFIG["risk_per_trade_pct"])]
            environment = {**os.environ, "PYTHONIOENCODING": "utf-8",
                           "PYTHONUTF8": "1"}
            completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                       encoding="utf-8", errors="replace", env=environment)
        if completed.returncode != 0:
            raise RuntimeError(f"Modal ensemble failed with exit code {completed.returncode}")
    else:
        from macro_forecast_engine.data_loader import fetch_market_data
        rows: list[dict[str, Any]] = []
        with _status("Running local CPU strategy ensemble..."):
            for timeframe in ("15m", "1h", "4h", "1D"):
                period, interval = TIMEFRAME_FETCH[timeframe]
                market = fetch_market_data(period, interval=interval)
                market_prices = market[[column for column in (
                    "gold_open", "gold_high", "gold_low", "gold_close"
                ) if column in market]]
                prices = normalize_timeframe_prices(market_prices, timeframe)
                matrix = run_strategy_matrix(
                    prices, list(STRATEGIES), [timeframe], INDICATOR_PARAMS,
                    CONFIG["account_equity"], CONFIG["risk_per_trade_pct"],
                )
                for row in matrix.to_dict("records"):
                    row["confidence"] = strategy_confidence(row)
                    rows.append(row)
        rows.sort(key=lambda row: (row["confidence"], row["net_expected_return"]), reverse=True)
        if not rows:
            raise ValueError("Local ensemble did not produce any strategy results")
        path.write_text(
            json.dumps(build_ensemble_payload(rows, "Local CPU"), indent=2, allow_nan=False),
            encoding="utf-8",
        )
    if not path.exists():
        raise FileNotFoundError(f"Ensemble artifact was not created: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Ensemble artifact could not be parsed: {path} ({error})") from error
    report = _as_mapping(report)
    leaderboard = report.get("leaderboard")
    if not isinstance(leaderboard, list) or not leaderboard:
        raise ValueError("Ensemble artifact does not contain a strategy leaderboard")
    environment = CONFIG["execution_environment"]
    table_title = (
        "Modal A10G Top 5 Strategy Leaderboard"
        if environment == "modal"
        else "Local CPU Top 5 Strategy Leaderboard"
    )
    table = Table(title=table_title, box=box.ASCII)
    table.add_column("Rank", justify="right")
    table.add_column("Strategy / Timeframe", style="green")
    table.add_column("Win Rate %", justify="right")
    table.add_column("Confidence %", justify="right")
    table.add_column("Net Return %", justify="right")
    for rank, raw_row in enumerate(leaderboard[:5], 1):
        row = _as_mapping(raw_row)
        table.add_row(str(rank),
                      f"{row.get('strategy', 'unknown')} / {row.get('timeframe', 'n/a')}",
                      f"{_to_float(row.get('win_rate')):.2f}",
                      f"{_to_float(row.get('confidence')):.2f}",
                      f"{_to_float(row.get('net_expected_return')):.2f}")
    console.print(table)
    console.print(Panel(f"[bold green]Recommended strategy: {report.get('recommendation', 'n/a')}[/bold green]",
                        border_style="green"))
    console.print(
        f"[green]Account equity:[/green] ${CONFIG['account_equity']:,.2f} | "
        f"[green]Risk per trade:[/green] {CONFIG['risk_per_trade_pct']:.2f}%"
    )


def build_ensemble_payload(rows: list[dict[str, Any]], environment: str) -> dict[str, Any]:
    """Create a lightweight, JSON-safe leaderboard artifact."""
    if not rows:
        raise ValueError("Cannot build an ensemble payload without strategy rows")
    ranked = [
        {key: value for key, value in row.items() if key != "closed_trades"}
        for row in rows[:5]
    ]
    return _json_safe({
        "execution_environment": environment,
        "generated_at_utc": pd.Timestamp.now(tz="UTC"),
        "leaderboard": ranked,
        "recommendation": f"{rows[0].get('strategy', 'unknown')} / {rows[0].get('timeframe', 'n/a')}",
    })


def setup_guide() -> None:
    readme = ROOT / "README.md"
    console.print(Markdown(readme.read_text(encoding="utf-8")))


def settings_menu() -> None:
    global CONFIG
    refresh_config()
    while True:
        current_engine = "Modal Cloud GPU" if CONFIG["execution_environment"] == "modal" else "Local Machine"
        choice = choose("FINRIX System Settings", [
            f"[1] Toggle Execution Engine (Current: {current_engine})",
            f"[2] Set Default Account Equity (${CONFIG['account_equity']:,.2f})",
            f"[3] Set Default Risk Per Trade ({CONFIG['risk_per_trade_pct']:.2f}%)",
            "[4] Save & Apply Settings (Return to Main Menu)",
        ])
        if choice.startswith("[1]"):
            CONFIG = save_config(CONFIG_PATH, {
                "execution_environment": "local"
                if CONFIG["execution_environment"] == "modal" else "modal",
            })
        elif choice.startswith("[2]"):
            try:
                equity = float(ask("Default account equity ($)", str(CONFIG["account_equity"])))
                if equity < 0:
                    raise ValueError
                CONFIG = save_config(CONFIG_PATH, {"account_equity": equity})
            except ValueError:
                console.print("[red][ERR] Equity must be a non-negative number.[/red]")
        elif choice.startswith("[3]"):
            try:
                risk = float(ask("Default risk per trade (%)", str(CONFIG["risk_per_trade_pct"])))
                if risk < 0:
                    raise ValueError
                CONFIG = save_config(CONFIG_PATH, {"risk_per_trade_pct": risk})
            except ValueError:
                console.print("[red][ERR] Risk must be a non-negative number.[/red]")
        else:
            CONFIG = save_config(CONFIG_PATH, CONFIG)
            return


def main() -> None:
    banner()
    choices = [
        "[1] Today's Best Trade Signal (Instant Execution Setup)",
        "[2] Daily Price Path & Signals (30-Day Forward Forecast)",
        "[3] Monthly Macro Horizon (22-Day Outlook & Weekly Pivots)",
        "[4] Yearly Regime Trend Shift (252-Day Forecast)",
        "[5] Macro News & FinBERT Sentiment Analysis",
        "[6] Trade Execution & Dynamic Levels Lookup",
        "[7] Historical Strategy Backtester",
        "[8] Multi-Strategy Ensemble Analyzer",
        "[9] System Settings",
        "[10] Setup & README Guide",
        "[11] Exit Terminal",
    ]
    while True:
        choice = choose("Select terminal operation", choices)
        try:
            clear_screen()
            if choice.startswith("[1]"):
                try:
                    today_signal()
                except Exception as error:
                    console.print(f"[bold red]Execution Error:[/bold red] {error}")
            elif choice.startswith("[2]"):
                show_forecast(run_forecast("daily"))
            elif choice.startswith("[3]"):
                show_forecast(run_forecast("monthly"))
            elif choice.startswith("[4]"):
                show_forecast(run_forecast("yearly"))
            elif choice.startswith("[5]"):
                news_report()
            elif choice.startswith("[6]"):
                execution_lookup()
            elif choice.startswith("[7]"):
                backtester()
            elif choice.startswith("[8]"):
                ensemble()
            elif choice.startswith("[9]"):
                settings_menu()
                banner()
            elif choice.startswith("[10]"):
                setup_guide()
            else:
                console.print("[green][OK] Terminal session closed.[/green]")
                return
        except Exception as error:
            console.print(Panel(f"[ERR] {error}", title="Operation Failed",
                                border_style="red", box=box.ASCII))
        if questionary is not None:
            questionary.press_any_key_to_continue("Press any key to return to the menu").ask()


if __name__ == "__main__":
    main()
