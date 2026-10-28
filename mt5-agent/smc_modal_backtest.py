"""Vectorized SMC/ICT XAUUSD backtesting with optional Modal parallelism.

Run locally:
    python smc_modal_backtest.py --data data/XAUUSD_M15.csv

Run on Modal:
    modal run smc_modal_backtest.py --data data/XAUUSD_M15.csv

For a genuine three-year M15 study, provide a three-year M15 CSV from a
broker/MT5 data source. Yahoo Finance does not reliably expose that history.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt
except ImportError as exc:  # pragma: no cover
    vbt = None
    _VECTORBT_ERROR = exc

try:
    import modal

    MODAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    modal = None
    MODAL_AVAILABLE = False


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "XAUUSD_M15.csv"
DEFAULT_OUTPUT = ROOT / "data" / "top_smc_gold_strategies.json"
START_CASH = 10_000.0
APP_NAME = "xauusd-smc-ict-backtester"

if MODAL_AVAILABLE:
    MODAL_IMAGE = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("numpy", "pandas", "vectorbt")
    )
    app = modal.App(APP_NAME)
else:  # pragma: no cover
    MODAL_IMAGE = None
    app = None


def _require_vectorbt() -> None:
    if vbt is None:
        raise RuntimeError(f"vectorbt is required: {_VECTORBT_ERROR}")


def load_ohlcv(path: Path | str) -> pd.DataFrame:
    """Load broker/MT5 OHLCV data and normalize column names."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Data file does not exist: {source}")
    frame = pd.read_csv(source)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    aliases = {"datetime": "time", "date": "time", "timestamp": "time"}
    frame = frame.rename(columns=aliases)
    required = {"time", "open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["time"]).set_index("time")
    frame.index = frame.index.tz_localize(None)
    numeric = ["open", "high", "low", "close", "volume"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame[numeric].dropna().sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def _rsi(close: pd.Series, length: int = 21) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return (100.0 - 100.0 / (1.0 + gain / loss.replace(0.0, np.nan))).rename("rsi")


def smc_features(
    data: pd.DataFrame,
    swing_period: int,
    mitigation_bars: int,
) -> Dict[str, pd.Series]:
    """Build SMC/ICT features using vectorized pandas operations."""
    high, low, close = data["high"], data["low"], data["close"]
    bullish_fvg = low > high.shift(2)
    bearish_fvg = high < low.shift(2)
    swing_high = high.shift(1).rolling(swing_period, min_periods=swing_period).max()
    swing_low = low.shift(1).rolling(swing_period, min_periods=swing_period).min()
    bos_up = close > swing_high
    bos_down = close < swing_low

    bull_bottom = high.shift(2).where(bullish_fvg)
    bull_top = low.where(bullish_fvg)
    bear_bottom = high.where(bearish_fvg)
    bear_top = low.shift(2).where(bearish_fvg)
    last_bull_bottom = bull_bottom.ffill(limit=mitigation_bars)
    last_bull_top = bull_top.ffill(limit=mitigation_bars)
    last_bear_bottom = bear_bottom.ffill(limit=mitigation_bars)
    last_bear_top = bear_top.ffill(limit=mitigation_bars)

    bullish_retest = (
        last_bull_bottom.notna()
        & (low <= last_bull_top)
        & (close >= last_bull_bottom)
    )
    bearish_retest = (
        last_bear_top.notna()
        & (high >= last_bear_bottom)
        & (close <= last_bear_top)
    )
    return {
        "bullish_fvg": bullish_fvg,
        "bearish_fvg": bearish_fvg,
        "bos_up": bos_up,
        "bos_down": bos_down,
        "bullish_retest": bullish_retest,
        "bearish_retest": bearish_retest,
    }


def smc_signal(data: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
    features = smc_features(
        data,
        swing_period=int(params["swing_period"]),
        mitigation_bars=int(params["mitigation_bars"]),
    )
    close = data["close"]
    ema = close.ewm(span=int(params["ema_period"]), adjust=False, min_periods=int(params["ema_period"])).mean()
    rsi = _rsi(close, int(params["rsi_period"]))
    rsi_long = rsi >= float(params["rsi_long_min"])
    rsi_short = rsi <= float(params["rsi_short_max"])
    long_signal = (
        features["bullish_retest"]
        & features["bos_up"]
        & (close > ema)
        & rsi_long
    )
    short_signal = (
        features["bearish_retest"]
        & features["bos_down"]
        & (close < ema)
        & rsi_short
    )
    signal = pd.Series(0, index=data.index, dtype=np.int8)
    signal.loc[long_signal] = 1
    signal.loc[short_signal] = -1
    return signal.rename("smc_signal")


def retail_signal(data: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
    """Retail baseline: RSI threshold entries plus EMA trend direction."""
    close = data["close"]
    ema = close.ewm(span=int(params["ema_period"]), adjust=False, min_periods=int(params["ema_period"])).mean()
    rsi = _rsi(close, int(params["rsi_period"]))
    signal = pd.Series(0, index=data.index, dtype=np.int8)
    signal.loc[(close > ema) & (rsi >= params["rsi_long_min"])] = 1
    signal.loc[(close < ema) & (rsi <= params["rsi_short_max"])] = -1
    return signal.rename("retail_signal")


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def backtest(data: pd.DataFrame, signal: pd.Series) -> Dict[str, float]:
    _require_vectorbt()
    close = data["close"].reindex(signal.index).astype(float)
    signal = signal.reindex(close.index).fillna(0).astype(int)
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=signal.eq(1),
        exits=signal.eq(-1),
        short_entries=signal.eq(-1),
        short_exits=signal.eq(1),
        init_cash=START_CASH,
        fees=0.0005,
        slippage=0.0002,
        upon_opposite_entry="close",
        accumulate=False,
        freq="15min",
    )
    try:
        trades = portfolio.trades.records_readable
        pnl = pd.to_numeric(trades["PnL"], errors="coerce").dropna()
    except (AttributeError, KeyError, TypeError, ValueError):
        pnl = pd.Series(dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_loss = abs(float(losses.sum()))
    gross_win = float(wins.sum())
    pf = gross_win / gross_loss if gross_loss else (999.0 if gross_win else 0.0)
    try:
        drawdown = abs(_finite(portfolio.max_drawdown()))
        sharpe = _finite(portfolio.sharpe_ratio())
        total_return = _finite(portfolio.total_return()) * 100.0
    except (AttributeError, TypeError, ValueError):
        drawdown, sharpe, total_return = 0.0, 0.0, 0.0
    return {
        "trades": int(len(pnl)),
        "profit_factor": round(min(pf, 999.0), 6),
        "win_rate": round(100.0 * len(wins) / len(pnl), 6) if len(pnl) else 0.0,
        "max_drawdown": round(drawdown * 100.0 if drawdown <= 1 else drawdown, 6),
        "sharpe_ratio": round(sharpe, 6),
        "total_return": round(total_return, 6),
    }


def parameter_grid() -> List[Dict[str, Any]]:
    common = [
        {"swing_period": swing, "mitigation_bars": bars, "rsi_long_min": 50, "rsi_short_max": 50}
        for swing in (5, 10, 20)
        for bars in (3, 8, 15)
    ]
    return [
        {
            **item,
            "ema_period": ema,
            "rsi_period": 21,
        }
        for item in common
        for ema in (100, 200)
    ]


def _evaluate_params(data: pd.DataFrame, params: Mapping[str, Any]) -> Dict[str, Any]:
    split = int(len(data) * 0.70)
    signal = smc_signal(data, params)
    is_metrics = backtest(data.iloc[:split], signal.iloc[:split])
    oos_metrics = backtest(data.iloc[split:], signal.iloc[split:])
    score = (
        oos_metrics["profit_factor"]
        + 0.25 * oos_metrics["sharpe_ratio"]
        - 0.05 * oos_metrics["max_drawdown"]
    )
    return {
        "strategy": "smc_ict_fvg_bos_retest",
        "parameters": dict(params),
        "metrics": {"is": is_metrics, "out_of_sample": oos_metrics, "score": round(score, 6)},
    }


if MODAL_AVAILABLE:

    @app.function(image=MODAL_IMAGE, timeout=900, retries=1)
    def evaluate_worker(data_json: str, params: Dict[str, Any]) -> Dict[str, Any]:
        data = pd.read_json(io.StringIO(data_json), orient="split")
        data.index = pd.to_datetime(data.index)
        return _evaluate_params(data, params)


def evaluate_grid(data: pd.DataFrame, use_modal: bool) -> List[Dict[str, Any]]:
    candidates = parameter_grid()
    if use_modal:
        if not MODAL_AVAILABLE:
            raise RuntimeError("Modal is not installed; use --local or install modal")
        data_json = data.to_json(orient="split", date_format="iso")
        results = list(evaluate_worker.starmap((data_json, params) for params in candidates))
    else:
        results = [_evaluate_params(data, params) for params in candidates]
    return sorted(
        [
            item
            for item in results
            if item["metrics"]["out_of_sample"]["max_drawdown"] < 20.0
        ],
        key=lambda item: (
            item["metrics"]["out_of_sample"]["profit_factor"],
            item["metrics"]["out_of_sample"]["sharpe_ratio"],
            -item["metrics"]["out_of_sample"]["max_drawdown"],
        ),
        reverse=True,
    )


def make_comparison(data: pd.DataFrame, winner: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    split = int(len(data) * 0.70)
    retail_params = {
        "ema_period": 200,
        "rsi_period": 21,
        "rsi_long_min": 50,
        "rsi_short_max": 50,
    }
    retail_signal_series = retail_signal(data, retail_params)
    retail = backtest(data.iloc[split:], retail_signal_series.iloc[split:])
    smc = winner["metrics"]["out_of_sample"] if winner else {
        "profit_factor": 0.0,
        "win_rate": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "trades": 0,
    }
    return {
        "SMC/ICT": {
            "parameters": winner["parameters"] if winner else {},
            "out_of_sample": smc,
        },
        "Retail EMA/RSI": {
            "parameters": retail_params,
            "out_of_sample": retail,
        },
    }


def export_results(
    data: pd.DataFrame,
    ranked: List[Dict[str, Any]],
    output: Path | str,
) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    winner = ranked[0] if ranked else None
    payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "data": {
            "rows": len(data),
            "start": data.index[0].isoformat(),
            "end": data.index[-1].isoformat(),
            "timeframe": "M15",
            "split": "70% IS / 30% OOS chronological",
        },
        "ranking": "OOS Profit Factor descending, then Sharpe descending, then Max Drawdown ascending",
        "max_oos_drawdown_filter_percent": 20.0,
        "strategies": ranked[:10],
        "comparison": make_comparison(data, winner),
    }
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return destination


def run(data_path: Path, output: Path, use_modal: bool) -> Path:
    data = load_ohlcv(data_path)
    if len(data) < 500:
        raise ValueError("SMC M15 research requires at least 500 clean bars")
    ranked = evaluate_grid(data, use_modal=use_modal)
    destination = export_results(data, ranked, output)
    print(f"Evaluated {len(data)} M15 bars with {len(parameter_grid())} workers.")
    print(f"Exported {len(ranked[:10])} qualifying strategies to {destination}")
    if ranked:
        print(json.dumps(ranked[0], indent=2))
    return destination


if MODAL_AVAILABLE:

    @app.local_entrypoint()
    def modal_main(
        data: str = str(DEFAULT_DATA),
        output: str = str(DEFAULT_OUTPUT),
        local: bool = False,
    ) -> None:
        run(Path(data), Path(output), use_modal=not local)

def cli_main() -> None:
    parser = argparse.ArgumentParser(description="SMC/ICT XAUUSD M15 backtester")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--local", action="store_true", help="Run workers in this Python process")
    args = parser.parse_args()
    run(Path(args.data), Path(args.output), use_modal=False if args.local else MODAL_AVAILABLE)


if __name__ == "__main__":
    cli_main()
