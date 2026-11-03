"""Interactive Rich terminal for the XAUUSD macro forecast engine."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
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

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from macro_forecast_engine.data_loader import fetch_live_news
from macro_forecast_engine.data_processor import analyze_news_sentiment
from macro_forecast_engine.trader_execution import build_execution_plan
from macro_forecast_engine.trader_risk import size_position
from macro_forecast_engine.trader_timemachine import (
    STRATEGIES, TIMEFRAME_FETCH, normalize_timeframe_prices, run_strategy_matrix,
    replay_forecasts,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
console = Console()


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
    clear_screen()
    art = Text(
        "+--------------------------------------------------------------+\n"
        "|     INSTITUTIONAL QUANT TRADING TERMINAL (XAUUSD)          |\n"
        "+--------------------------------------------------------------+",
        style="bold green",
    )
    console.print(Panel(art, border_style="green", padding=(1, 2)))
    status = Table.grid(padding=(0, 2))
    status.add_column(style="green")
    status.add_column()
    status.add_row("SYSTEM", "[bold green]ONLINE[/bold green]")
    status.add_row("MODAL GPU", "[bold green]A10G worker connected on demand[/bold green]")
    status.add_row("ACTIVE SKILLS", "TFT | SMC | FinBERT | Kelly | Volatility Parity | Timemachine")
    status.add_row("DATA ROOT", str(DATA_DIR))
    console.print(Panel(status, title="Terminal Status", border_style="green"))


def artifact_path(mode: str) -> Path:
    filename = {"monthly": "macro_month_forecast.json", "yearly": "macro_year_forecast.json"}.get(
        mode, f"macro_{mode}_forecast.json"
    )
    return DATA_DIR / filename


def run_forecast(mode: str, force: bool = True) -> dict[str, Any]:
    path = artifact_path(mode)
    if not force and path.exists():
        age_seconds = pd.Timestamp.now().timestamp() - path.stat().st_mtime
        if age_seconds < 900:
            return json.loads(path.read_text(encoding="utf-8"))
    command = [sys.executable, "-m", "modal", "run", "macro_forecast_engine/modal_engine.py",
               "--mode", mode]
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with console.status(f"[green]Running {mode} forecast on Modal A10G...[/green]"):
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                   encoding="utf-8", errors="replace", env=environment)
    if completed.returncode != 0:
        console.print(Panel("[ERR] " + completed.stdout + completed.stderr,
                            title="Modal Error", border_style="red", box=box.ASCII))
        raise RuntimeError(f"Modal forecast failed with exit code {completed.returncode}")
    if not path.exists():
        raise FileNotFoundError(f"Forecast artifact was not created: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def print_execution_card(payload: dict[str, Any], equity: float) -> None:
    risk = payload["risk_levels"]
    sizing = payload["position_sizing"]
    bias = payload["directional_bias"]
    action = "BUY" if bias == "bullish" else "SELL" if bias == "bearish" else "NO TRADE (Wait for Zone)"
    table = Table(title="Today's Best Trade - Instant Execution Card", box=box.ASCII)
    table.add_column("Metric", style="green")
    table.add_column("Value", style="bold white")
    table.add_row("Recommended Action", action)
    table.add_row("Ideal Entry Zone", str(risk["entry_zone"]))
    table.add_row("Dynamic SL", f"${risk['stop_loss']:,.4f}")
    table.add_row("TP1 / TP2 / TP3", " / ".join(f"${risk[key]:,.4f}" for key in
                                                ("take_profit_1", "take_profit_2", "take_profit_3")))
    table.add_row("Kelly Position Size", f"{sizing['units']:.6f} units (equity ${equity:,.2f})")
    table.add_row("Confidence", f"{payload['confidence_score']:.2f}%")
    reward_ratio = float(risk.get("risk_reward", sizing.get("reward_risk", 0.0)))
    table.add_row("Risk : Reward", f"1 : {reward_ratio:.2f}")
    console.print(table)


def show_forecast(payload: dict[str, Any]) -> None:
    console.print(Panel(
        f"[bold green]{payload['mode'].capitalize()}[/bold green] | "
        f"{payload['data']['horizon']} business days | "
        f"Bias: {payload['directional_bias'].upper()} "
        f"({payload['confidence_score']:.2f}%)\n"
        f"Target: ${payload['monte_carlo']['target_price']:,.4f}  "
        f"Corridor: ${payload['monte_carlo']['target_low']:,.4f} - "
        f"${payload['monte_carlo']['target_high']:,.4f}\n"
        f"SMC Buy: {payload['buy_zone']}  SMC Sell: {payload['sell_zone']}\n"
        f"SL: ${payload['risk_levels']['stop_loss']:,.4f}  "
        f"TP1/TP2/TP3: ${payload['risk_levels']['take_profit_1']:,.4f} / "
        f"${payload['risk_levels']['take_profit_2']:,.4f} / "
        f"${payload['risk_levels']['take_profit_3']:,.4f}",
        title="Forecast Overview",
        border_style="green", box=box.ASCII,
    ))
    table = Table(title="TFT Forward Path", box=box.ASCII)
    for column in ("Date", "q10", "q50", "q90", "Signal"):
        table.add_column(column, justify="right" if column != "Signal" else "left")
    for row in payload["forecast"]:
        table.add_row(row["date"], f"${row['expected_low_q10']:,.2f}",
                      f"${row['target_q50']:,.2f}", f"${row['expected_high_q90']:,.2f}",
                      row["signal_bias"].upper())
    console.print(table)


def today_signal() -> None:
    payload = run_forecast("daily", force=False)
    equity = float(ask("Account equity ($)", "100000"))
    entry_zone = payload["risk_levels"]["entry_zone"]
    entry = float(entry_zone[0] if payload["directional_bias"] == "bullish" else entry_zone[1])
    payload["risk_levels"] = build_execution_plan(
        entry, payload["forecast"], payload["smc"], payload["directional_bias"],
    )
    sizing = size_position(
        entry,
        payload["forecast"][0]["expected_low_q10"],
        payload["forecast"][0]["target_q50"],
        payload["forecast"][0]["expected_high_q90"],
        equity,
    )
    payload["position_sizing"] = sizing
    print_execution_card(payload, equity)


def news_report() -> None:
    with console.status("[green]Fetching macro news and FinBERT sentiment...[/green]"):
        news = fetch_live_news()
        sentiment_data = analyze_news_sentiment(news)
        sentiment = sentiment_data["score"]
    table = Table(title="Macro News & FinBERT", box=box.ROUNDED)
    table.add_column("Metric", style="green")
    table.add_column("Value")
    table.add_row("Headlines", str(len(news)))
    table.add_row("Hawkish / Dovish Score", f"{sentiment:+.4f}" if not pd.isna(sentiment) else "unavailable")
    table.add_row("Bias", "BULLISH" if sentiment > 0.05 else "BEARISH"
                  if sentiment < -0.05 else "NEUTRAL")
    table.add_row("Sentiment Velocity", f"{sentiment_data['velocity']:+.4f}"
                  if not pd.isna(sentiment_data["velocity"]) else "unavailable")
    console.print(table)
    if not news.empty:
        for title in news["headline"].head(8).tolist():
            console.print(f"  - {title}")
    else:
        console.print("  No gold macro headlines were returned by the approved feeds.")


def execution_lookup() -> None:
    payload = run_forecast("daily")
    price = float(ask("Current market price ($)", str(payload["forecast"][0]["target_q50"])))
    equity = float(ask("Account equity ($)", "100000"))
    risk_percent = float(ask("Risk per trade (%)", "1.0"))
    plan = build_execution_plan(price, payload["forecast"], payload["smc"], payload["directional_bias"])
    row = payload["forecast"][0]
    sizing = size_position(price, row["expected_low_q10"], row["target_q50"],
                           row["expected_high_q90"], equity,
                           risk_fraction=max(0.0, risk_percent) / 100.0)
    sizing["pip_risk"] = round(abs(price - plan["stop_loss"]) / 0.01, 2)
    payload["risk_levels"] = plan
    payload["position_sizing"] = sizing
    print_execution_card(payload, equity)


def backtester() -> None:
    strategies = choose_many("Select strategies to test", list(STRATEGIES))
    timeframes = choose_many("Select timeframes to test", ["1m", "5m", "15m", "1h", "4h", "1D"])
    console.print(
        f"[green]Selected strategies:[/green] {', '.join(strategies)}\n"
        f"[green]Selected timeframes:[/green] {', '.join(timeframes)}"
    )
    from macro_forecast_engine.data_loader import fetch_market_data
    with console.status("[green]Loading historical XAUUSD data...[/green]"):
        frames = []
        for timeframe in timeframes:
            period, interval = TIMEFRAME_FETCH[timeframe]
            market = fetch_market_data(period, interval=interval)
            frames.append((timeframe, normalize_timeframe_prices(market["gold_close"], timeframe)))
    result = pd.concat(
        [run_strategy_matrix(prices, strategies, [timeframe]) for timeframe, prices in frames],
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


def _historical_log_return_forecast(history: pd.Series, horizon: int) -> list[float]:
    """Use only the expanding history to produce a backend trend forecast."""
    returns = pd.Series(history).pct_change().dropna().tail(252)
    drift = float((returns + 1.0).clip(lower=1e-9).map(math.log).mean())
    return (float(history.iloc[-1]) * pd.Series(
        np.exp(np.cumsum([drift] * horizon))
    )).tolist()


def ensemble() -> None:
    proceed = ask(
        "[WARNING] Parallel multi-strategy backtesting on Modal A10G consumes cloud credits. "
        "Proceed? (Y/N)", "N",
    ).strip().lower()
    if proceed not in {"y", "yes"}:
        console.print("[yellow]Cancelled; no Modal jobs were submitted.[/yellow]")
        return
    with console.status("[green]Running parallel strategy ensemble on Modal A10G...[/green]"):
        command = [sys.executable, "-m", "modal", "run",
                   "macro_forecast_engine/modal_engine.py", "--mode", "ensemble"]
        environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                   encoding="utf-8", errors="replace", env=environment)
    if completed.returncode != 0:
        raise RuntimeError(f"Modal ensemble failed with exit code {completed.returncode}")
    path = DATA_DIR / "modal_ensemble.json"
    if not path.exists():
        raise FileNotFoundError(f"Ensemble artifact was not created: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    table = Table(title="Modal A10G Top 5 Strategy Leaderboard", box=box.ASCII)
    table.add_column("Rank", justify="right")
    table.add_column("Strategy / Timeframe", style="green")
    table.add_column("Win Rate %", justify="right")
    table.add_column("Confidence %", justify="right")
    table.add_column("Net Return %", justify="right")
    for rank, row in enumerate(report["leaderboard"][:5], 1):
        table.add_row(str(rank), f"{row['strategy']} / {row['timeframe']}",
                      f"{row['win_rate']:.2f}", f"{row['confidence']:.2f}",
                      f"{row['net_expected_return']:.2f}")
    console.print(table)
    console.print(Panel(f"[bold green]Recommended strategy: {report['recommendation']}[/bold green]",
                        border_style="green"))


def setup_guide() -> None:
    readme = ROOT / "README.md"
    console.print(Markdown(readme.read_text(encoding="utf-8")))


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
        "[9] Setup & README Guide",
        "[10] Exit Terminal",
    ]
    while True:
        choice = choose("Select terminal operation", choices)
        try:
            clear_screen()
            if choice.startswith("[1]"):
                today_signal()
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
                setup_guide()
            else:
                console.print("[green][OK] Terminal session closed.[/green]")
                return
        except (ValueError, FileNotFoundError, RuntimeError, OSError) as error:
            console.print(Panel(f"[ERR] {error}", title="Operation Failed",
                                border_style="red", box=box.ASCII))
        if questionary is not None:
            questionary.press_any_key_to_continue("Press any key to return to the menu").ask()


if __name__ == "__main__":
    main()
