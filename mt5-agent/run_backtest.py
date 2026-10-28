"""Step 1b: Run backtests with realistic costs and IS/OOS validation.

Usage:
    python run_backtest.py                        # both strategies, EURUSD H1
    python run_backtest.py --strategy smc --symbol XAUUSD --tf M15
    python run_backtest.py --sample               # force bundled sample data

Read-only: never connects a trading session, never sends orders.
"""
import argparse
import os
import sys

from backtesting import Backtest

import config
from mt5_data import load_ohlcv, split_is_oos
from strategies import STRATEGIES
from strategies_v2 import STRATEGIES2

ALL_STRATEGIES = {**STRATEGIES, **STRATEGIES2}


def _get(stats, *keys, default=float("nan")):
    """Fetch a stats key tolerantly across backtesting.py versions."""
    for k in keys:
        try:
            v = stats[k]
            if v is not None:
                return v
        except Exception:
            continue
    return default


def trade_metrics(trades) -> dict:
    """Compute Win Rate / Profit Factor / Expectancy from the trades table."""
    if trades is None or len(trades) == 0:
        return {"trades": 0, "win_rate": float("nan"), "profit_factor": float("nan"),
                "expectancy": float("nan"), "avg_win": 0.0, "avg_loss": 0.0}
    pnl = trades["PnL"].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    return {
        "trades": len(pnl),
        "win_rate": 100.0 * len(wins) / len(pnl),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "expectancy": pnl.mean(),          # avg cash PnL per trade
        "avg_win": wins.mean() if len(wins) else 0.0,
        "avg_loss": losses.mean() if len(losses) else 0.0,
    }


def run_one(bt_name: str, strat_cls, df, spread, label, save_dir):
    bt = Backtest(df, strat_cls,
                  cash=config.START_CASH,
                  spread=spread,
                  commission=config.COMMISSION,
                  margin=config.MARGIN,
                  exclusive_orders=True,
                  finalize_trades=True)
    stats = bt.run()
    m = trade_metrics(getattr(stats, "_trades", None))
    result = {
        "label": label,
        "return_pct": _get(stats, "Return [%]"),
        "bh_return_pct": _get(stats, "Buy & Hold Return [%]"),
        "max_dd_pct": _get(stats, "Max. Drawdown [%]", "Max Drawdown [%]"),
        "sharpe": _get(stats, "Sharpe Ratio"),
        "exposure_pct": _get(stats, "Exposure Time [%]"),
        **m,
    }
    # persist trade + equity logs for auditing
    os.makedirs(save_dir, exist_ok=True)
    stem = f"{bt_name}_{label.replace(' ', '').replace(':', '_')}"
    trades = getattr(stats, "_trades", None)
    if trades is not None and len(trades):
        trades.to_csv(os.path.join(save_dir, f"{stem}_trades.csv"))
    eq = getattr(stats, "_equity_curve", None)
    if eq is not None:
        eq.to_csv(os.path.join(save_dir, f"{stem}_equity.csv"))
    return result


def fmt(v, pct=False, money=False):
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    if pct or money:
        return f"{v:,.2f}{'%' if pct else ''}"
    return f"{v:,.2f}"


def print_table(rows):
    hdr = f"{'segment':<10} {'trades':>6} {'win%':>7} {'PF':>6} {'expect$':>9} " \
          f" {'return%':>8} {'MaxDD%':>8} {'Sharpe':>7} {'OOS?':>4}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:<10} {r['trades']:>6} {fmt(r['win_rate']):>7} "
              f"{fmt(r['profit_factor']):>6} {fmt(r['expectancy'], money=True):>9} "
              f" {fmt(r['return_pct'], pct=True):>8} {fmt(r['max_dd_pct'], pct=True):>8} "
              f"{fmt(r['sharpe']):>7} {r.get('oos', ''):>4}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["ema", "smc", "smc2", "both", "all"],
                    default="both")
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--tf", default="H1", choices=list(config.TIMEFRAMES))
    ap.add_argument("--sample", action="store_true", help="force bundled sample data")
    args = ap.parse_args()

    if args.sample:
        from backtesting.test import EURUSD
        df = EURUSD[["Open", "High", "Low", "Close", "Volume"]].copy()
        source = "bundled sample (forced via --sample)"
    else:
        df, source = load_ohlcv(args.symbol, args.tf)
    print(f"Data: {source}")
    print(f"Costs: spread={config.SPREAD:.5f} (relative), "
          f"commission={config.COMMISSION:.5f}, cash=${config.START_CASH:,.0f}")
    print(f"Split: IS first {config.OOS_SPLIT:.0%} / OOS last {1-config.OOS_SPLIT:.0%}\n")

    is_df, oos_df = split_is_oos(df, config.OOS_SPLIT)
    names = {"both": ["ema", "smc"], "all": list(ALL_STRATEGIES)}.get(
        args.strategy, [args.strategy])
    save_dir = os.path.join(str(config.BASE_DIR), "results", f"{args.symbol}_{args.tf}")

    for name in names:
        cls = ALL_STRATEGIES[name]
        print(f"=== {name.upper()} : {cls.__name__} ===")
        rows = [
            run_one(name, cls, is_df, config.SPREAD, "IS", save_dir),
            run_one(name, cls, oos_df, config.SPREAD, "OOS", save_dir),
            run_one(name, cls, df, config.SPREAD, "FULL", save_dir),
        ]
        rows[1]["oos"] = "<--"   # OOS = untouched validation data
        print_table(rows)
        oos_pf, oos_n = rows[1]["profit_factor"], rows[1]["trades"]
        verdict = ("OOS too few trades - inconclusive" if oos_n < 30 else
                   "OOS PF >= 1.1 -> promising, needs walk-forward + more data" if oos_pf >= 1.1 else
                   "OOS PF < 1.1 -> NO EDGE on unseen data (as-is)")
        print(f"Verdict: {verdict}")
        print(f"Logs: {save_dir}\\{name}_*_trades.csv / *_equity.csv\n")
    print("Reminder: backtest != live. Next gates: MT5 Strategy Tester (MQL5), "
          "then >=2-4 weeks demo forward-test.")


if __name__ == "__main__":
    sys.exit(main())
