"""Gold strategy research engine.

The module is intentionally read-only: it downloads market data, evaluates
technical signals with vectorbt, and writes a ranked JSON research report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - depends on the local environment
    yf = None
    _YFINANCE_IMPORT_ERROR = exc

try:
    import vectorbt as vbt
except ImportError as exc:  # pragma: no cover - depends on the local environment
    vbt = None
    _VECTORBT_IMPORT_ERROR = exc

try:
    import pandas_ta as ta
except ImportError:  # pandas_ta is optional because the engine has safe fallbacks.
    ta = None


DEFAULT_PERIOD = "730d"
DEFAULT_INTERVAL = "1h"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "top_gold_strategies.json"


def _require_dependency(module: Any, name: str, error: Optional[Exception]) -> None:
    if module is None:
        detail = f": {error}" if error is not None else ""
        raise RuntimeError(f"{name} is required for this operation{detail}")


def _as_series(value: Any, index: pd.Index, name: str = "") -> pd.Series:
    """Convert pandas-ta/vectorbt-like outputs to one aligned numeric series."""
    if isinstance(value, pd.DataFrame):
        if name in value.columns:
            value = value[name]
        elif len(value.columns) == 1:
            value = value.iloc[:, 0]
        else:
            raise ValueError(f"Cannot select a series from columns {list(value.columns)}")
    if not isinstance(value, pd.Series):
        value = pd.Series(value, index=index)
    return pd.to_numeric(value.reindex(index), errors="coerce")


def _first_column(frame: pd.DataFrame, prefixes: Iterable[str]) -> Optional[pd.Series]:
    for prefix in prefixes:
        matches = [column for column in frame.columns if str(column).upper().startswith(prefix)]
        if matches:
            return frame[matches[0]]
    return None


def fetch_gold_data(
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    min_rows: int = 100,
) -> pd.DataFrame:
    """Download and normalize OHLCV data for gold futures or spot gold."""
    _require_dependency(yf, "yfinance", globals().get("_YFINANCE_IMPORT_ERROR"))
    last_error: Optional[Exception] = None
    for ticker in ("GC=F", "XAUUSD=X"):
        try:
            raw = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if raw is None or raw.empty:
                raise ValueError("the data provider returned no rows")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.rename(columns={str(column): str(column).title() for column in raw.columns})
            required = ["Open", "High", "Low", "Close"]
            missing = [column for column in required if column not in raw.columns]
            if missing:
                raise ValueError(f"missing OHLC columns: {missing}")
            if "Volume" not in raw.columns:
                raw["Volume"] = 0.0
            data = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            data.columns = ["open", "high", "low", "close", "volume"]
            data.index = pd.to_datetime(data.index, errors="coerce", utc=True)
            data.index = data.index.tz_localize(None)
            data = data[~data.index.isna()].sort_index()
            data = data[~data.index.duplicated(keep="last")].dropna()
            data = data.astype(float)
            if len(data) >= min_rows:
                return data
            raise ValueError(f"only {len(data)} usable rows returned")
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not fetch gold data from GC=F or XAUUSD=X: {last_error}")


class IndicatorEngine:
    """Indicator and signal factory with pandas-ta-compatible fallbacks."""

    def __init__(self, data: pd.DataFrame):
        required = {"open", "high", "low", "close"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"data is missing columns: {sorted(missing)}")
        self.data = data.copy().sort_index().dropna(subset=sorted(required))
        self.close = self.data["close"].astype(float)
        self.high = self.data["high"].astype(float)
        self.low = self.data["low"].astype(float)

    def ema(self, length: int) -> pd.Series:
        length = max(1, int(length))
        if ta is not None:
            result = ta.ema(self.close, length=length)
            if result is not None:
                return _as_series(result, self.data.index).rename("ema")
        return self.close.ewm(span=length, adjust=False, min_periods=length).mean().rename("ema")

    def rsi(self, **kwargs: Any) -> pd.Series:
        length = int(kwargs.get("rsi_length", kwargs.get("length", 14)))
        length = max(1, length)
        if ta is not None:
            result = ta.rsi(self.close, length=length)
            if result is not None:
                return _as_series(result, self.data.index).rename("rsi")
        delta = self.close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
        rs = gain / loss.replace(0.0, np.nan)
        return (100.0 - 100.0 / (1.0 + rs)).rename("rsi")

    def atr(self, **kwargs: Any) -> pd.Series:
        length = max(1, int(kwargs.get("atr_length", kwargs.get("length", 14))))
        if ta is not None:
            result = ta.atr(self.high, self.low, self.close, length=length)
            if result is not None:
                return _as_series(result, self.data.index).rename("atr")
        previous_close = self.close.shift(1)
        true_range = pd.concat(
            [
                self.high - self.low,
                (self.high - previous_close).abs(),
                (self.low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.ewm(alpha=1 / length, adjust=False, min_periods=length).mean().rename("atr")

    def macd(self, **kwargs: Any) -> Tuple[pd.Series, pd.Series, pd.Series]:
        fast = max(1, int(kwargs.get("macd_fast", kwargs.get("fast", 12))))
        slow = max(fast + 1, int(kwargs.get("macd_slow", kwargs.get("slow", 26))))
        signal_length = max(1, int(kwargs.get("macd_signal", kwargs.get("signal", 9))))
        if ta is not None:
            result = ta.macd(self.close, fast=fast, slow=slow, signal=signal_length)
            if isinstance(result, pd.DataFrame) and not result.empty:
                line = _first_column(result, (f"MACD_{fast}_{slow}_{signal_length}", "MACD_"))
                signal = _first_column(result, (f"MACDs_{fast}_{slow}_{signal_length}", "MACDs_"))
                histogram = _first_column(result, (f"MACDh_{fast}_{slow}_{signal_length}", "MACDh_"))
                if line is not None and signal is not None:
                    return (
                        _as_series(line, self.data.index, name="macd"),
                        _as_series(signal, self.data.index, name="signal"),
                        _as_series(histogram, self.data.index, name="histogram")
                        if histogram is not None
                        else (_as_series(line, self.data.index) - _as_series(signal, self.data.index)),
                    )
        line = self.close.ewm(span=fast, adjust=False, min_periods=fast).mean()
        line = line - self.close.ewm(span=slow, adjust=False, min_periods=slow).mean()
        signal = line.ewm(span=signal_length, adjust=False, min_periods=signal_length).mean()
        return line.rename("macd"), signal.rename("signal"), (line - signal).rename("histogram")

    def supertrend(self, **kwargs: Any) -> Tuple[pd.Series, pd.Series]:
        length = max(1, int(kwargs.get("supertrend_length", kwargs.get("length", 10))))
        multiplier = float(kwargs.get("supertrend_multiplier", kwargs.get("multiplier", 3.0)))
        atr = self.atr(length=length)
        midpoint = (self.high + self.low) / 2.0
        upper = midpoint + multiplier * atr
        lower = midpoint - multiplier * atr
        trend = pd.Series(index=self.data.index, dtype=float)
        direction = pd.Series(1.0, index=self.data.index)
        for i in range(len(self.data)):
            if i == 0:
                trend.iloc[i] = upper.iloc[i]
                continue
            previous = trend.iloc[i - 1]
            if self.close.iloc[i] > previous:
                direction.iloc[i] = 1.0
            elif self.close.iloc[i] < previous:
                direction.iloc[i] = -1.0
            else:
                direction.iloc[i] = direction.iloc[i - 1]
            trend.iloc[i] = lower.iloc[i] if direction.iloc[i] > 0 else upper.iloc[i]
        return trend.rename("supertrend"), direction.rename("direction")

    def signal(self, strategy: str, **kwargs: Any) -> pd.Series:
        strategy = strategy.lower().strip()
        signal = pd.Series(0, index=self.data.index, dtype=int)
        if strategy == "ema_cross":
            fast = int(kwargs.get("ema_fast", kwargs.get("fast", 12)))
            slow = int(kwargs.get("ema_slow", kwargs.get("slow", 26)))
            if fast >= slow:
                raise ValueError(f"ema_fast ({fast}) must be less than ema_slow ({slow})")
            fast_line, slow_line = self.ema(fast), self.ema(slow)
            signal[fast_line > slow_line] = 1
            signal[fast_line < slow_line] = -1
        elif strategy == "rsi":
            values = self.rsi(**kwargs)
            overbought = float(kwargs.get("overbought", 70))
            oversold = float(kwargs.get("oversold", 30))
            signal[values < oversold] = 1
            signal[values > overbought] = -1
        elif strategy == "macd":
            line, trigger, _ = self.macd(**kwargs)
            signal[line > trigger] = 1
            signal[line < trigger] = -1
        elif strategy == "atr_breakout":
            atr = self.atr(**kwargs)
            lookback = max(1, int(kwargs.get("breakout_length", kwargs.get("lookback", 20))))
            multiplier = float(kwargs.get("breakout_multiplier", kwargs.get("multiplier", 1.0)))
            upper = self.high.rolling(lookback).max().shift(1) + multiplier * atr
            lower = self.low.rolling(lookback).min().shift(1) - multiplier * atr
            signal[self.close > upper] = 1
            signal[self.close < lower] = -1
        elif strategy == "supertrend":
            _, direction = self.supertrend(**kwargs)
            signal[direction > 0] = 1
            signal[direction < 0] = -1
        else:
            raise ValueError(f"Unsupported strategy family: {strategy}")
        return signal.fillna(0).astype(int).rename("signal")


def _metric_value(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _portfolio_metrics(portfolio: Any) -> Dict[str, float]:
    try:
        records = portfolio.trades.records_readable
        pnl = pd.to_numeric(records["PnL"], errors="coerce").dropna()
    except (AttributeError, TypeError, ValueError):
        pnl = pd.Series(dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_loss = abs(float(losses.sum()))
    gross_win = float(wins.sum())
    profit_factor = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    win_rate = 100.0 * len(wins) / len(pnl) if len(pnl) else 0.0
    try:
        sharpe = _metric_value(portfolio.sharpe_ratio())
    except (AttributeError, TypeError, ValueError):
        sharpe = 0.0
    try:
        max_drawdown = abs(_metric_value(portfolio.max_drawdown()))
    except (AttributeError, TypeError, ValueError):
        max_drawdown = 0.0
    return {
        "trades": int(len(pnl)),
        "profit_factor": round(min(profit_factor, 999.0), 6),
        "sharpe_ratio": round(sharpe, 6),
        "max_drawdown": round(max_drawdown * 100.0 if max_drawdown <= 1.0 else max_drawdown, 6),
        "win_rate": round(win_rate, 6),
        "total_return": round(_metric_value(portfolio.total_return()) * 100.0, 6),
    }


def run_vectorbt_backtest(
    data: pd.DataFrame,
    signal: pd.Series,
    allow_short: bool = True,
    init_cash: float = 10_000.0,
    fees: float = 0.0005,
    slippage: float = 0.0002,
) -> Dict[str, Any]:
    """Run vectorbt with explicit long/short boolean masks."""
    _require_dependency(vbt, "vectorbt", globals().get("_VECTORBT_IMPORT_ERROR"))
    close = pd.to_numeric(data["close"], errors="coerce").reindex(signal.index).dropna()
    signal = pd.to_numeric(signal, errors="coerce").reindex(close.index).fillna(0).astype(int)
    entries = signal.eq(1)
    exits = signal.eq(0) if not allow_short else signal.eq(-1)
    short_entries = signal.eq(-1) if allow_short else None
    short_exits = signal.eq(1) if allow_short else None
    portfolio = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=float(init_cash),
        fees=float(fees),
        slippage=float(slippage),
        upon_opposite_entry="close",
        accumulate=False,
        freq=None,
    )
    return {"portfolio": portfolio, "metrics": _portfolio_metrics(portfolio)}


def _parameter_grid() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "ema_cross": [
            {"ema_fast": fast, "ema_slow": slow}
            for fast in (8, 12, 20)
            for slow in (26, 40, 60)
            if fast < slow
        ],
        "rsi": [
            {"rsi_length": length, "oversold": oversold, "overbought": overbought}
            for length in (7, 14, 21)
            for oversold, overbought in ((25, 70), (30, 70), (30, 75))
        ],
        "macd": [
            {"macd_fast": fast, "macd_slow": slow, "macd_signal": signal}
            for fast, slow, signal in ((8, 21, 5), (12, 26, 9), (19, 39, 9))
        ],
        "atr_breakout": [
            {"atr_length": length, "breakout_length": lookback, "breakout_multiplier": multiplier}
            for length in (14, 21)
            for lookback in (20, 40)
            for multiplier in (0.5, 1.0)
        ],
        "supertrend": [
            {"supertrend_length": length, "supertrend_multiplier": multiplier}
            for length in (7, 10, 14)
            for multiplier in (2.0, 3.0, 4.0)
        ],
    }


def _safe_stability(os_pf: float, is_pf: float) -> float:
    if is_pf <= 0:
        return 0.0
    return os_pf / is_pf


def grid_search(
    data: pd.DataFrame,
    allow_short: bool = True,
    parameter_grid: Optional[Mapping[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate all families using a chronological 70/30 split."""
    if len(data) < 20:
        raise ValueError("At least 20 clean bars are required for a 70/30 split")
    split = max(1, min(len(data) - 1, int(len(data) * 0.70)))
    is_data, os_data = data.iloc[:split], data.iloc[split:]
    grid = parameter_grid or _parameter_grid()
    results: List[Dict[str, Any]] = []
    for family, candidates in grid.items():
        for params in candidates:
            engine = IndicatorEngine(data)
            try:
                full_signal = engine.signal(family, **params)
                is_result = run_vectorbt_backtest(
                    is_data, full_signal.iloc[:split], allow_short=allow_short
                )
                os_result = run_vectorbt_backtest(
                    os_data, full_signal.iloc[split:], allow_short=allow_short
                )
            except (KeyError, TypeError, ValueError, RuntimeError):
                continue
            is_metrics = is_result["metrics"]
            os_metrics = os_result["metrics"]
            stability = _safe_stability(os_metrics["profit_factor"], is_metrics["profit_factor"])
            score = (
                os_metrics["sharpe_ratio"] * 0.4
                + os_metrics["profit_factor"] * 0.4
                + stability * 0.2
            )
            if (
                os_metrics["profit_factor"] > 1.5
                and os_metrics["win_rate"] > 55.0
                and os_metrics["max_drawdown"] < 30.0
                and stability >= 0.75
            ):
                results.append(
                    {
                        "strategy": family,
                        "parameters": dict(params),
                        "metrics": {
                            "is": is_metrics,
                            "out_of_sample": os_metrics,
                            "stability": round(stability, 6),
                            "score": round(score, 6),
                        },
                    }
                )
    return sorted(results, key=lambda item: item["metrics"]["score"], reverse=True)


def export_top_strategies(
    strategies: List[Dict[str, Any]],
    output_path: Path | str = DEFAULT_OUTPUT,
    limit: int = 10,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "strategy_count": min(limit, len(strategies)),
        "strategies": strategies[:limit],
    }
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Grid-search gold strategies with vectorbt.")
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, choices=("1h", "1d"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--long-only", action="store_true")
    args = parser.parse_args()
    data = fetch_gold_data(period=args.period, interval=args.interval)
    strategies = grid_search(data, allow_short=not args.long_only)
    output = export_top_strategies(strategies, args.output)
    print(f"Evaluated {len(data)} bars; exported {len(strategies[:10])} strategies to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
