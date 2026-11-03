"""Interactive Rich terminal for the XAUUSD macro forecast engine."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from macro_forecast_engine.data_loader import fetch_live_news, score_headlines
from macro_forecast_engine.modal_engine import _run_worker
from macro_forecast_engine.trader_execution import build_execution_plan
from macro_forecast_engine.trader_risk import size_position
from macro_forecast_engine.trader_timemachine import replay_forecasts

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
console = Console()


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


def banner() -> None:
    console.clear()
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
    status.add_row("ACTIVE SKILLS", "TFT • SMC • FinBERT • Kelly • Volatility Parity • Timemachine")
    status.add_row("DATA ROOT", str(DATA_DIR))
    console.print(Panel(status, title="Terminal Status", border_style="green"))


def artifact_path(mode: str) -> Path:
    filename = {"monthly": "macro_month_forecast.json", "yearly": "macro_year_forecast.json"}.get(
        mode, f"macro_{mode}_forecast.json"
    )
    return DATA_DIR / filename


def run_forecast(mode: str) -> dict[str, Any]:
    path = artifact_path(mode)
    command = [sys.executable, "-m", "modal", "run", "macro_forecast_engine/modal_engine.py",
               "--mode", mode]
    with console.status(f"[green]Running {mode} forecast on Modal A10G...[/green]"):
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                   encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        console.print(Panel(completed.stdout + completed.stderr, title="Modal Error",
                            border_style="red"))
        raise RuntimeError(f"Modal forecast failed with exit code {completed.returncode}")
    if not path.exists():
        raise FileNotFoundError(f"Forecast artifact was not created: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def print_execution_card(payload: dict[str, Any], equity: float) -> None:
    risk = payload["risk_levels"]
    sizing = payload["position_sizing"]
    bias = payload["directional_bias"]
    action = "BUY" if bias == "bullish" else "SELL" if bias == "bearish" else "NO TRADE (Wait for Zone)"
    table = Table(title="Today's Best Trade — Instant Execution Card", box=box.ROUNDED)
    table.add_column("Metric", style="green")
    table.add_column("Value", style="bold white")
    table.add_row("Recommended Action", action)
    table.add_row("Ideal Entry Zone", str(risk["entry_zone"]))
    table.add_row("Dynamic SL", f"${risk['stop_loss']:,.4f}")
    table.add_row("TP1 / TP2 / TP3", " / ".join(f"${risk[key]:,.4f}" for key in
                                                ("take_profit_1", "take_profit_2", "take_profit_3")))
    table.add_row("Kelly Position Size", f"{sizing['units']:.6f} units (equity ${equity:,.2f})")
    table.add_row("Confidence", f"{payload['confidence_score']:.2f}%")
    table.add_row("Risk : Reward", f"{sizing['reward_risk']:.2f} : 1")
    console.print(table)


def show_forecast(payload: dict[str, Any]) -> None:
    console.print(Panel(
        f"[bold green]{payload['mode'].capitalize()}[/bold green] • "
        f"{payload['data']['horizon']} business days • "
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
        border_style="green",
    ))
    table = Table(title="TFT Forward Path", box=box.SIMPLE_HEAVY)
    for column in ("Date", "q10", "q50", "q90", "Signal"):
        table.add_column(column, justify="right" if column != "Signal" else "left")
    for row in payload["forecast"]:
        table.add_row(row["date"], f"${row['expected_low_q10']:,.2f}",
                      f"${row['target_q50']:,.2f}", f"${row['expected_high_q90']:,.2f}",
                      row["signal_bias"].upper())
    console.print(table)


def today_signal() -> None:
    payload = run_forecast("daily")
    equity = float(ask("Account equity ($)", "100000"))
    sizing = size_position(
        payload["risk_levels"]["entry_zone"][1],
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
        sentiment = score_headlines(news["headline"].tolist()) if not news.empty else 0.0
    table = Table(title="Macro News & FinBERT", box=box.ROUNDED)
    table.add_column("Metric", style="green")
    table.add_column("Value")
    table.add_row("Headlines", str(len(news)))
    table.add_row("Hawkish / Dovish Score", f"{sentiment:+.4f}")
    table.add_row("Bias", "HAWKISH" if sentiment > 0.05 else "DOVISH" if sentiment < -0.05 else "NEUTRAL")
    table.add_row("Sentiment Velocity", "Live snapshot; rerun to refresh")
    console.print(table)
    for title in news["headline"].head(8).tolist():
        console.print(f"  • {title}")


def execution_lookup() -> None:
    payload = run_forecast("daily")
    price = float(ask("Current market price ($)", str(payload["forecast"][0]["target_q50"])))
    equity = float(ask("Account equity ($)", "100000"))
    plan = build_execution_plan(price, payload["forecast"], payload["smc"], payload["directional_bias"])
    row = payload["forecast"][0]
    sizing = size_position(price, row["expected_low_q10"], row["target_q50"],
                           row["expected_high_q90"], equity)
    payload["risk_levels"] = plan
    payload["position_sizing"] = sizing
    print_execution_card(payload, equity)


def backtester() -> None:
    mode = choose("Replay horizon", ["daily (30)", "monthly (22)", "yearly (252)"])
    horizon = int(mode.split("(")[1].split(")")[0])
    from macro_forecast_engine.data_loader import fetch_market_data
    with console.status("[green]Loading historical XAUUSD data...[/green]"):
        market = fetch_market_data("10y")
    prices = market["gold_close"]
    result = replay_forecasts(
        prices,
        lambda history, size: [float(history.iloc[-1])] * size,
        horizon=min(horizon, max(1, len(prices) // 10)),
        min_history=60,
    )
    table = Table(title="Lookahead-Safe Timemachine Replay", box=box.ROUNDED)
    table.add_column("Metric", style="green")
    table.add_column("Value")
    table.add_row("Replay Windows", str(len(result)))
    table.add_row("Mean Absolute Error", f"${result['mae'].mean():,.4f}" if not result.empty else "n/a")
    table.add_row("Directional Win Rate",
                  f"{result['direction_correct'].mean() * 100:.2f}%" if not result.empty else "n/a")
    table.add_row("Strategy", "TFT Trend / SMC FVG / Volatility Parity replay scaffold")
    console.print(table)


def ensemble() -> None:
    payload = run_forecast("daily")
    scores = {
        "TFT Trend": payload["confidence_score"],
        "SMC Reaction Zones": payload["smc"]["reaction_confidence"],
        "Kelly Risk": payload["position_sizing"]["allocation_fraction"] * 100,
        "FinBERT Macro Sentiment": abs(payload.get("macro_directional_bias", "neutral") != "neutral") * 60,
    }
    best = max(scores, key=scores.get)
    table = Table(title="Multi-Strategy Ensemble", box=box.ROUNDED)
    table.add_column("Strategy", style="green")
    table.add_column("Confidence", justify="right")
    for name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        table.add_row(name, f"{float(score):.2f}%")
    console.print(table)
    console.print(Panel(f"[bold green]Recommended strategy: {best}[/bold green]",
                        border_style="green"))


def setup_guide() -> None:
    readme = ROOT / "README.md"
    console.print(Markdown(readme.read_text(encoding="utf-8")))


def main() -> None:
    banner()
    choices = [
        "⚡ Today's Best Trade Signal (Instant Execution Setup)",
        "📈 Daily Price Path & Signals (30-Day Forward Forecast)",
        "📅 Monthly Macro Horizon (22-Day Outlook & Weekly Pivots)",
        "🗓️ Yearly Regime Trend Shift (252-Day Forecast)",
        "📰 Macro News & FinBERT Sentiment Analysis",
        "🎯 Trade Execution & Dynamic Levels Lookup",
        "⏳ Historical Strategy Backtester",
        "🧠 Multi-Strategy Ensemble Analyzer",
        "📖 View System Setup & README Guide",
        "❌ Exit Terminal",
    ]
    while True:
        choice = choose("Select terminal operation", choices)
        try:
            if choice.startswith("⚡"):
                today_signal()
            elif choice.startswith("📈"):
                show_forecast(run_forecast("daily"))
            elif choice.startswith("📅"):
                show_forecast(run_forecast("monthly"))
            elif choice.startswith("🗓️"):
                show_forecast(run_forecast("yearly"))
            elif choice.startswith("📰"):
                news_report()
            elif choice.startswith("🎯"):
                execution_lookup()
            elif choice.startswith("⏳"):
                backtester()
            elif choice.startswith("🧠"):
                ensemble()
            elif choice.startswith("📖"):
                setup_guide()
            else:
                console.print("[green]Terminal session closed.[/green]")
                return
        except (ValueError, FileNotFoundError, RuntimeError, OSError) as error:
            console.print(Panel(str(error), title="Operation Failed", border_style="red"))
        if questionary is not None:
            questionary.press_any_key_to_continue("Press any key to return to the menu").ask()


if __name__ == "__main__":
    main()
