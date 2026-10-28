"""Walk-forward optimization for EMACrossATR (no look-ahead, honest OOS).

Rolling-origin validation: optimize on ALL history before a fold's test
window, apply chosen params to the unseen window. One fold per invocation
(fits 30s command budgets); --summarize aggregates fold JSONs.

    python -u walk_forward.py --symbol XAUUSD --tf H1 --folds 3 --fold 1
    python -u walk_forward.py --symbol XAUUSD --tf H1 --folds 3 --fold 2
    ...
    python -u walk_forward.py --symbol XAUUSD --tf H1 --folds 3 --summarize
"""
import argparse
import json
import multiprocessing
import os
import sys

os.environ.setdefault("TQDM_DISABLE", "1")   # silence progress bars

import backtesting as btmod                   # noqa: E402

# TRUE multi-process optimization (thread-based thrashes the GIL on Windows)
btmod.Pool = lambda *a, **k: multiprocessing.Pool(4)

from backtesting import Backtest              # noqa: E402

import config                                 # noqa: E402
from mt5_data import load_ohlcv               # noqa: E402
from strategies import EMACrossATR            # noqa: E402
from run_backtest import trade_metrics, _get  # noqa: E402

GRID_FULL = dict(fast=[10, 20, 30], slow=[50, 100, 150],
                 sl_atr=[1.0, 1.5, 2.0], tp_atr=[2.0, 3.0, 4.0])      # 81 combos
GRID_QUICK = dict(fast=[15, 30], slow=[60, 120],
                  sl_atr=[1.5], tp_atr=[2.0, 3.0])                    # 8 combos
MIN_OOS_TRADES = 15
PF_PASS = 1.2


def make_bt(df):
    return Backtest(df, EMACrossATR,
                    cash=config.START_CASH, spread=config.SPREAD,
                    commission=config.COMMISSION, margin=config.MARGIN,
                    exclusive_orders=True, finalize_trades=True)


def fold_metrics(stats):
    m = trade_metrics(getattr(stats, "_trades", None))
    m["return_pct"] = _get(stats, "Return [%]")
    m["max_dd_pct"] = _get(stats, "Max. Drawdown [%]", "Max Drawdown [%]")
    m["sqn"] = _get(stats, "SQN")
    return m


def best_params_from(stats, heatmap, grid):
    """Extract best-run params from the optimization heatmap (0.6.x)."""
    hm = heatmap.copy()
    if isinstance(hm.index, pd.MultiIndex) or not all(p in hm.columns for p in grid):
        hm = hm.reset_index()
    score_col = [c for c in hm.columns if str(c) in ("SQN", "Return [%]")][0]
    row = hm.loc[hm[score_col].idxmax()]
    return {p: int(row[p]) if float(row[p]).is_integer() else float(row[p])
            for p in grid}


import pandas as pd  # noqa: E402


def fold_bounds(n, folds, k):
    """Test window [start, end) for fold k (0-based), expanding train before."""
    test_size = n // (folds + 1)
    test_end = n - (folds - 1 - k) * test_size
    return test_end - test_size, test_end


def run_fold(symbol, tf, folds, k, quick):
    grid = GRID_QUICK if quick else GRID_FULL
    df, src = load_ohlcv(symbol, tf)
    n = len(df)
    s, e = fold_bounds(n, folds, k)
    train_df, test_df = df.iloc[:s], df.iloc[s:e]
    n_combos = len(grid["fast"]) * len(grid["slow"]) * len(grid["sl_atr"]) * len(grid["tp_atr"])
    print(f"Data: {src}\nFold {k + 1}/{folds}: {n_combos} combos, maximize=SQN")
    print(f"  train: {len(train_df)} bars {train_df.index[0].date()}..{train_df.index[-1].date()}")
    print(f"  test:  {len(test_df)} bars {test_df.index[0].date()}..{test_df.index[-1].date()}")
    sys.stdout.reconfigure(line_buffering=True)

    stats, heatmap = make_bt(train_df).optimize(
        maximize="SQN", method="grid", return_heatmap=True, **grid)
    params = best_params_from(stats, heatmap, grid)
    print(f"  best IS: {params} (IS SQN={float(stats['SQN']):.2f}, "
          f"IS trades={int(stats['# Trades'])})")

    st = make_bt(test_df).run(**params)
    m = fold_metrics(st)
    passed = m["trades"] >= MIN_OOS_TRADES and m["profit_factor"] >= PF_PASS
    m.update({"fold": k + 1, "folds": folds, "symbol": symbol, "tf": tf,
              "params": params,
              "train_span": f"{train_df.index[0]}..{train_df.index[-1]}",
              "test_span": f"{test_df.index[0]}..{test_df.index[-1]}",
              "pass": bool(passed)})
    out = os.path.join(str(config.BASE_DIR), "results",
                       f"wf_{symbol}_{tf}_fold{k + 1}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(m, f, indent=2, default=str)
    print(f"  OOS: trades={m['trades']} win%={m['win_rate']:.1f} PF={m['profit_factor']:.2f} "
          f"expect={m['expectancy']:.2f} return={m['return_pct']:.1f}% "
          f"MaxDD={m['max_dd_pct']:.1f}% SQN={m['sqn']:.2f} -> "
          f"{'PASS' if passed else 'FAIL'}")
    print(f"Saved: {out}")


def summarize(symbol, tf, folds):
    rows = []
    for k in range(folds):
        path = os.path.join(str(config.BASE_DIR), "results",
                            f"wf_{symbol}_{tf}_fold{k + 1}.json")
        if not os.path.exists(path):
            print(f"[WARN] missing {path} - run --fold {k + 1} first")
            continue
        with open(path) as f:
            rows.append(json.load(f))
    if not rows:
        print("[ERROR] no fold results found")
        return 1
    print(f"\nWalk-forward: {symbol} {tf} ({len(rows)} folds)")
    print(f"{'fold':>4} {'IS params':<42} {'OOS trades':>10} {'win%':>6} "
          f"{'PF':>5} {'expect$':>8} {'MaxDD%':>7} {'SQN':>5} {'pass':>5}")
    for r in rows:
        p = r["params"]
        tag = "f%s/s%s/sl%s/tp%s" % (p["fast"], p["slow"], p["sl_atr"], p["tp_atr"])
        print(f"{r['fold']:>4} "
              f"{tag:<42} "
              f"{r['trades']:>10} {r['win_rate']:>6.1f} {r['profit_factor']:>5.2f} "
              f"{r['expectancy']:>8.2f} {r['max_dd_pct']:>7.1f} {r['sqn']:>5.2f} "
              f"{'YES' if r['pass'] else 'no':>5}")
    pfs = [r["profit_factor"] for r in rows
           if r["profit_factor"] == r["profit_factor"]]
    n_pass = sum(bool(r["pass"]) for r in rows)
    stable = len({(r["params"]["fast"], r["params"]["slow"]) for r in rows}) == 1
    print(f"folds passing PF>={PF_PASS} & trades>={MIN_OOS_TRADES}: "
          f"{n_pass}/{len(rows)} | OOS PF mean={sum(pfs)/len(pfs):.2f} "
          f"min={min(pfs):.2f} max={max(pfs):.2f} | params_stable={stable}")
    if n_pass == len(rows) and stable:
        verdict = "PROCEED -> candidate for MQL5 + demo forward-test"
    elif n_pass >= 1:
        verdict = ("PARTIAL -> some folds pass but unstable = overfitting risk; "
                   "NOT live-ready")
    else:
        verdict = "REJECT -> no robust OOS edge in this parameter region"
    print(f"VERDICT: {verdict}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--tf", default="H1", choices=list(config.TIMEFRAMES))
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--fold", type=int, default=None,
                    help="run a single fold (1-based)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        return summarize(args.symbol, args.tf, args.folds)
    if args.fold is None:
        print("Specify --fold N (or --summarize)")
        return 2
    run_fold(args.symbol, args.tf, args.folds, args.fold - 1, args.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())


