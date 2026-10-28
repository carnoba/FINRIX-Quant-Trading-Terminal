"""strategy_miner.py - automated strategy discovery & validation loop (background).

Searches four UNT-TESTED families on real MT5 history:
  A  volbreak  Donchian/Keltner volatility breakout + expansion filter
  B  session   Asia-range liquidity sweep -> London/NY expansion
  C  mtf       H1/H4 trend regime + M15/M5 entries in high-vol hours only
  D  meanrev   RSI extreme + Bollinger fade, ATR stops, BB-mid target

VALIDATION PROTOCOL (anti-overfit):
  1. Chronological 70/30 IS/OOS split; grid on IS, ONE look at OOS.
  2. Expanding walk-forward, 3 folds (train ends 40/55/70%).
  3. Realistic costs: full config.SPREAD on entry, config.COMMISSION
     round-trip, 1% risk sizing, 40x leverage cap, pessimistic same-bar
     SL-first, gap-through-SL exits at open, forced flat at EOF.

PASS GATES (ALL required):
  OOS     : trades>=30, PF>=1.3, win>=40%, expectancy>0
  walk-fwd: aggregate PF>=1.3, trades>=60, >=2/3 folds PF>=1.0, worst>=0.7
  stabl.  : OOS PF >= 0.7 x IS PF; 3/3 distinct param sets = flagged unstable
A PASS only earns an MQL5 Strategy-Tester slot - never straight to live.

USAGE: python strategy_miner.py [--quick] [--families meanrev session ...]
Output: results/winner_strategies.json + results/miner_log.txt
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
from mt5_data import load_ohlcv
from strategies import atr_wilder, ema

sys.stdout.reconfigure(line_buffering=True)

WIN_MIN = 40.0
OOS_PF_MIN = 1.3
WF_PF_MIN = 1.3
MIN_OOS_TRADES = 30
MIN_WF_TRADES = 60
DEGRADE_MIN = 0.7
FOLD_PF_MIN = 0.7
RISK_PCT = 1.0
WARMUP = 60


def rsi(close, length):
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    ru = pd.Series(up).ewm(alpha=1 / length, adjust=False).mean()
    rd = pd.Series(dn).ewm(alpha=1 / length, adjust=False).mean()
    rs = ru / rd.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs.to_numpy())
    return np.where(np.isnan(out), 50.0, out)


def metrics_from_trades(trades):
    if not trades:
        return {"trades": 0, "win": 0.0, "pf": 0.0, "expect": 0.0,
                "maxdd": 0.0, "avgr": 0.0, "pnl": 0.0}
    pnl = np.array([t["pnl"] for t in trades], dtype=float)
    r = np.array([t["r"] for t in trades], dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gw, gl = wins.sum(), abs(losses.sum())
    pf = gw / gl if gl > 0 else (999.0 if gw > 0 else 0.0)
    eq = config.START_CASH + np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    dd = float(np.max((peak - eq) / peak) * 100.0)
    return {"trades": len(pnl), "win": 100.0 * len(wins) / len(pnl),
            "pf": min(pf, 999.0), "expect": float(pnl.mean()),
            "maxdd": dd, "avgr": float(r.mean()), "pnl": float(pnl.sum())}


def run_engine(df, d, sl, tp, exl, exs, max_hold):
    """Replay one param set; arrays already sliced to df.
    Fills at next bar open with full spread on the entry side; same-bar
    SL-before-TP; gap-through exits at the open; forced flat at EOF."""
    o = df["Open"].to_numpy()
    h = df["High"].to_numpy()
    l = df["Low"].to_numpy()
    c = df["Close"].to_numpy()
    n = len(df)
    equity = config.START_CASH
    trades = []
    pos = None
    for i in range(n):
        if pos is None and i >= 1 and d[i - 1] != 0:
            sd = int(d[i - 1])
            entry = o[i] * (1.0 + config.SPREAD) if sd > 0 \
                else o[i] * (1.0 - config.SPREAD)
            stop, target = sl[i - 1], tp[i - 1]
            risk = (entry - stop) if sd > 0 else (stop - entry)
            ok = (0 < risk) and ((target > entry) if sd > 0
                                 else (target < entry))
            if ok:
                units = equity * RISK_PCT / 100.0 / risk
                units = min(units, equity * config.MAX_LEVERAGE / entry)
                if units > 0:
                    pos = {"dir": sd, "entry": entry, "sl": stop, "tp": target,
                           "units": units, "fill_i": i, "risk": risk}
        if pos is not None:
            sd, stop, target = pos["dir"], pos["sl"], pos["tp"]
            px = reason = None
            if sd > 0:
                if l[i] <= stop:
                    px, reason = (o[i] if o[i] <= stop else stop), "SL"
                elif h[i] >= target:
                    px, reason = (o[i] if o[i] >= target else target), "TP"
            else:
                if h[i] >= stop:
                    px, reason = (o[i] if o[i] >= stop else stop), "SL"
                elif l[i] <= target:
                    px, reason = (o[i] if o[i] <= target else target), "TP"
            if px is None and (exl[i] if sd > 0 else exs[i]):
                px, reason = c[i], "EXIT"
            if px is None and max_hold and (i - pos["fill_i"]) >= max_hold:
                px, reason = c[i], "TIME"
            if px is None and i == n - 1:
                px, reason = c[i], "EOF"
            if px is not None:
                pnl = sd * (px - pos["entry"]) * pos["units"]
                pnl -= config.COMMISSION * pos["entry"] * pos["units"]
                equity += pnl
                trades.append({"i": i, "pnl": pnl,
                               "r": sd * (px - pos["entry"]) / pos["risk"],
                               "reason": reason})
                pos = None
    return trades, metrics_from_trades(trades)


# ---------------------------------------------------------------------------
# signal families: each returns (d, sl, tp, exl, exs, max_hold) arrays
# computed on the FULL df once per param set (no look-ahead: signals valid
# at bar i close -> engine fills at bar i+1 open)
# ---------------------------------------------------------------------------
def sig_volbreak(df, p):
    o, h, l, c = (df[x].to_numpy() for x in ("Open", "High", "Low", "Close"))
    a = atr_wilder(h, l, c, 14)
    n = len(df)
    d = np.zeros(n); sl = np.full(n, np.nan); tp = np.full(n, np.nan)
    if p["chan"] == "donchian":
        k = p["k"]
        dc_up = pd.Series(h).rolling(k).max().shift(1).to_numpy()
        dc_dn = pd.Series(l).rolling(k).min().shift(1).to_numpy()
        brk_up, brk_dn = c > dc_up, c < dc_dn
    else:  # keltner
        mid = ema(c, p["k"])
        band = p["mult"] * a
        brk_up, brk_dn = c > mid + band, c < mid - band
    # expansion filter: ATR now must exceed its own average (energy present)
    exp_avg = pd.Series(a).rolling(50).mean().to_numpy()
    expanded = a > p["exp"] * exp_avg
    long_i = np.where(brk_up & expanded)[0]
    short_i = np.where(brk_dn & expanded)[0]
    for i in long_i:
        d[i] = 1; sl[i] = c[i] - p["sl_atr"] * a[i]
        tp[i] = c[i] + p["rr"] * p["sl_atr"] * a[i]
    for i in short_i:
        d[i] = -1; sl[i] = c[i] + p["sl_atr"] * a[i]
        tp[i] = c[i] - p["rr"] * p["sl_atr"] * a[i]
    return d, sl, tp, np.zeros(n, bool), np.zeros(n, bool), 0


def sig_session(df, p):
    """Asia range (00-08 server) sweep -> London/NY reversal entry."""
    h, l, c = df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()
    a = atr_wilder(h, l, c, 14)
    hours = df.index.hour.to_numpy()
    day = df.index.date
    n = len(df)
    d = np.zeros(n); sl = np.full(n, np.nan); tp = np.full(n, np.nan)
    asia = (hours >= 0) & (hours < 8)
    ny = (hours >= 13) & (hours < 21)
    for dd in np.unique(day):
        day_i = np.where(day == dd)[0]
        asi = day_i[asia[day_i]]
        if len(asi) < 4:
            continue
        ah, al = h[asi].max(), l[asi].min()
        rng = ah - al
        for i in day_i[ny[day_i]]:     # only trade London/NY session
            if np.isnan(a[i]) or rng < p["min_atr"] * a[i]:
                continue
            if l[i] < al and c[i] > al:      # sweep Asia low -> reclaim = long
                d[i] = 1; sl[i] = al - p["buf"] * a[i]
                tp[i] = c[i] + p["rr"] * (c[i] - sl[i])
            elif h[i] > ah and c[i] < ah:    # sweep Asia high -> reject = short
                d[i] = -1; sl[i] = ah + p["buf"] * a[i]
                tp[i] = c[i] - p["rr"] * (sl[i] - c[i])
    return d, sl, tp, np.zeros(n, bool), np.zeros(n, bool), 12


def sig_mtf(df, p):
    """H1/H4 trend filter + M5/M15 trend-pullback entries in London/NY hours."""
    h, l, c = df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()
    a = atr_wilder(h, l, c, 14)
    hours = df.index.hour.to_numpy()
    n = len(df)
    # higher-TF EMA on COMPLETED htf bars, shifted -> no look-ahead
    tf = pd.Series(c, index=df.index)
    htf = tf.resample(p["htf"]).last().dropna()
    tr = htf.ewm(span=p["htf_ema"], adjust=False).mean().shift(1)
    idx = pd.Index(df.index)
    trend = tr.reindex(tr.index.union(idx)).sort_index().ffill().reindex(idx).to_numpy()
    ema_f, ema_s = ema(c, p["fast"]), ema(c, p["slow"])
    d = np.zeros(n); sl = np.full(n, np.nan); tp = np.full(n, np.nan)
    exl = np.zeros(n, bool); exs = np.zeros(n, bool)
    hvol = ((hours >= 8) & (hours < 17))       # London + NY hours only
    for i in range(WARMUP, n):
        if np.isnan(a[i]) or not hvol[i] or not np.isfinite(trend[i]):
            continue
        up = ema_f[i] > ema_s[i] and ema_f[i - 1] <= ema_s[i - 1]
        dn = ema_f[i] < ema_s[i] and ema_f[i - 1] >= ema_s[i - 1]
        if up and c[i] > trend[i]:
            d[i] = 1; sl[i] = c[i] - p["sl_atr"] * a[i]
            tp[i] = c[i] + p["rr"] * p["sl_atr"] * a[i]
        elif dn and c[i] < trend[i]:
            d[i] = -1; sl[i] = c[i] + p["sl_atr"] * a[i]
            tp[i] = c[i] - p["rr"] * p["sl_atr"] * a[i]
        if ema_f[i] < ema_s[i]:
            exl[i] = True
        if ema_f[i] > ema_s[i]:
            exs[i] = True
    return d, sl, tp, exl, exs, 0


def sig_meanrev(df, p):
    """Fade RSI extremes confirmed by Bollinger band touch; target = BB mid."""
    h, l, c = df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy()
    a = atr_wilder(h, l, c, 14)
    r = rsi(c, p["rsi_len"])
    s = pd.Series(c)
    mid = s.rolling(20).mean().to_numpy()
    sd = s.rolling(20).std().to_numpy()
    up_b, dn_b = mid + p["k"] * sd, mid - p["k"] * sd
    n = len(df)
    d = np.zeros(n); sl = np.full(n, np.nan); tp = np.full(n, np.nan)
    for i in range(WARMUP, n):
        if np.isnan(a[i]) or np.isnan(mid[i]):
            continue
        if r[i] < p["th"] and c[i] < dn_b[i]:        # oversold + lower band
            d[i] = 1; sl[i] = c[i] - p["sl_atr"] * a[i]; tp[i] = mid[i]
        elif r[i] > 100 - p["th"] and c[i] > up_b[i]:
            d[i] = -1; sl[i] = c[i] + p["sl_atr"] * a[i]; tp[i] = mid[i]
    return d, sl, tp, np.zeros(n, bool), np.zeros(n, bool), p["max_hold"]


# ---------------------------------------------------------------------------
# parameter grids + dataset matrix
# ---------------------------------------------------------------------------
def volbreak_grid():
    g = []
    for k in (20, 55):
        for exp in (1.0, 1.3):
            for sl_atr in (1.5, 3.0):
                for rr in (1.5, 2.5):
                    g.append({"chan": "donchian", "k": k, "exp": exp,
                              "sl_atr": sl_atr, "rr": rr})
    for k in (20, 50):
        for mult in (1.5, 2.5):
            for exp in (1.0, 1.3):
                g.append({"chan": "keltner", "k": k, "mult": mult,
                          "exp": exp, "sl_atr": 2.0, "rr": 2.0})
    return g


def session_grid():
    return [{"min_atr": ma, "buf": b, "rr": rr}
            for ma in (0.0, 1.0) for b in (0.3, 0.8)
            for rr in (1.5, 2.5)]


def mtf_grid():
    return [{"htf": htf, "htf_ema": te, "fast": f, "slow": s,
             "sl_atr": sa, "rr": rr}
            for htf, te in (("1h", 50), ("4h", 25))
            for f, s in ((5, 21), (9, 27))
            for sa in (1.5, 2.5) for rr in (1.5, 2.5)]


def meanrev_grid(quick=False):
    g = [{"rsi_len": rl, "th": th, "k": 2.0, "sl_atr": sa,
          "max_hold": mh, "rr": 0}
         for rl in (7, 14) for th in (20, 30)
         for sa in (1.5, 2.5) for mh in (24, 60)]
    return g[:8] if quick else g


FAMILIES = {
    "volbreak": (sig_volbreak, volbreak_grid,
                 [("XAUUSD", "M15"), ("XAUUSD", "H1"),
                  ("EURUSD", "M15"), ("GBPUSD", "H1")]),
    "session": (sig_session, session_grid,
                [("XAUUSD", "M15"), ("EURUSD", "M15"), ("GBPUSD", "M15")]),
    "mtf": (sig_mtf, mtf_grid,
            [("XAUUSD", "M15"), ("EURUSD", "M15"), ("GBPUSD", "M15")]),
    "meanrev": (sig_meanrev, meanrev_grid,
                [("XAUUSD", "H1"), ("EURUSD", "H1"), ("GBPUSD", "H1")]),
}
QUICK_DATASETS = {"volbreak": [("XAUUSD", "M15")],
                  "session": [("XAUUSD", "M15")],
                  "mtf": [("XAUUSD", "M15")],
                  "meanrev": [("XAUUSD", "H1")]}


# ---------------------------------------------------------------------------
# validation protocol
# ---------------------------------------------------------------------------
def eval_params(df, arrays, i0, i1):
    """Run engine on df[i0:i1] with full-data signal arrays sliced identically."""
    d, sl, tp, exl, exs, mh = arrays
    seg = df.iloc[i0:i1]
    return run_engine(seg, d[i0:i1], sl[i0:i1], tp[i0:i1],
                      exl[i0:i1], exs[i0:i1], mh)


def pick_best(cands):
    """(params, metrics, arrays) with best PF; requires >=30 trades, else >=10."""
    valid = [c for c in cands if c[1]["trades"] >= MIN_OOS_TRADES]
    if not valid:
        valid = [c for c in cands if c[1]["trades"] >= 10] or cands
    return max(valid, key=lambda c: (c[1]["pf"], c[1]["trades"]))


def gate(rec):
    """Apply pass gates to a record; mutate + return (passed, reasons)."""
    oos, ism = rec["oos"], rec["is"]
    wf = rec["walk_forward"]
    folds = wf["folds"]
    reasons = []
    if oos["trades"] < MIN_OOS_TRADES:
        reasons.append("OOS trades<30")
    if oos["pf"] < OOS_PF_MIN:
        reasons.append(f"OOS PF<{OOS_PF_MIN}")
    if oos["win"] < WIN_MIN:
        reasons.append(f"win<{WIN_MIN}%")
    if oos["expect"] <= 0:
        reasons.append("OOS expectancy<=0")
    if wf["aggregate"]["pf"] < WF_PF_MIN:
        reasons.append(f"WF PF<{WF_PF_MIN}")
    if wf["aggregate"]["trades"] < MIN_WF_TRADES:
        reasons.append("WF trades<60")
    ok_folds = sum(1 for f in folds if f["test_pf"] >= 1.0)
    if ok_folds < max(2, len(folds) - 1):
        reasons.append("too many folds PF<1")
    if folds and min(f["test_pf"] for f in folds) < FOLD_PF_MIN:
        reasons.append(f"fold PF<{FOLD_PF_MIN}")
    if ism["pf"] > 0 and oos["pf"] < DEGRADE_MIN * ism["pf"]:
        reasons.append("OOS degradation >30%")
    rec["reject_reasons"] = reasons
    rec["passed"] = not reasons
    return rec["passed"], reasons


def mine_family(fam, quick, log):
    sig_fn, grid_fn, datasets = FAMILIES[fam]
    grid = grid_fn(quick) if fam == "meanrev" else grid_fn()
    if quick:
        datasets = QUICK_DATASETS[fam]
    train_ends = (0.4,) if quick else (0.4, 0.55, 0.7)
    winners, results = [], []

    for symbol, tf in datasets:
        df, src = load_ohlcv(symbol, tf)
        n = len(df)
        cut = int(n * config.OOS_SPLIT)
        t0 = time.time()
        # ---- stage 1: IS grid -> single OOS look ------------------------
        scored = []
        for p in grid:
            arrays = sig_fn(df, p)
            _, is_m = eval_params(df, arrays, 0, cut)
            scored.append((p, is_m, arrays))
        best_p, best_is, best_arr = pick_best(scored)
        _, oos_m = eval_params(df, best_arr, cut, n)
        log(f"  [{symbol} {tf}] {len(grid)} grids | IS PF={best_is['pf']:.2f} "
            f"n={best_is['trades']} -> OOS PF={oos_m['pf']:.2f} "
            f"win={oos_m['win']:.1f}% n={oos_m['trades']} "
            f"({time.time()-t0:.1f}s)")
        # ---- stage 2: expanding walk-forward ---------------------------
        folds, wf_trades = [], []
        for f, tr_end_frac in enumerate(train_ends):
            tr_end = int(n * tr_end_frac)
            te_end = n if quick else int(n * (0.4 + 0.15 * (f + 1)))
            fcands = []
            for p in grid:
                arrays = sig_fn(df, p)
                _, m = eval_params(df, arrays, 0, tr_end)
                fcands.append((p, m, arrays))
            fp, _, farr = pick_best(fcands)
            ftr, fm = eval_params(df, farr, tr_end, te_end)
            folds.append({"train_end_frac": tr_end_frac, "params": fp,
                          "test_pf": fm["pf"], "test_trades": fm["trades"],
                          "test_win": fm["win"]})
            wf_trades.extend(ftr)
            log(f"    fold{f+1}: test PF={fm['pf']:.2f} n={fm['trades']} "
                f"params={fp}")
        wf_m = metrics_from_trades(wf_trades)
        uniq = len({json.dumps(f["params"], sort_keys=True) for f in folds})
        rec = {"family": fam, "symbol": symbol, "tf": tf, "bars": n,
               "params": best_p, "is": best_is, "oos": oos_m,
               "walk_forward": {"folds": folds, "aggregate": wf_m,
                                "distinct_param_sets": uniq,
                                "param_unstable": uniq == len(folds) and len(folds) > 1},
               "data_source": src}
        passed, reasons = gate(rec)
        results.append(rec)
        if passed:
            winners.append(rec)
            log(f"  *** {fam}/{symbol}/{tf} PASSED ALL GATES ***")
        else:
            log(f"  REJECT: {', '.join(reasons)}")
    return winners, results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Strategy discovery miner")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES),
                    choices=list(FAMILIES))
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 1 dataset/family, tiny grid, 1 fold")
    args = ap.parse_args()

    out_dir = str(config.BASE_DIR / "results")
    import os
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "miner_log.txt")
    win_path = os.path.join(out_dir, "winner_strategies.json")

    t_start = time.time()
    lines = []

    def log(msg):
        stamp = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}]"
        print(msg)
        lines.append(f"{stamp} {msg}")
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    log(f"strategy_miner START quick={args.quick} "
        f"families={args.families} costs: spread={config.SPREAD} "
        f"comm={config.COMMISSION} risk={RISK_PCT}%")
    log(f"gates: OOS PF>={OOS_PF_MIN} win>={WIN_MIN}% n>={MIN_OOS_TRADES} | "
        f"WF PF>={WF_PF_MIN} n>={MIN_WF_TRADES} | degrade<={1-DEGRADE_MIN:.0%}")

    all_winners, all_results = [], []
    for fam in args.families:
        log(f"=== FAMILY {fam} ===")
        w, r = mine_family(fam, args.quick, log)
        all_winners.extend(w)
        all_results.extend(r)
        # persist progressively so a crash loses nothing
        with open(win_path, "w", encoding="utf-8") as fh:
            json.dump({"updated_utc": datetime.now(timezone.utc).isoformat(),
                       "gates": {"oos_pf_min": OOS_PF_MIN,
                                 "win_min": WIN_MIN,
                                 "wf_pf_min": WF_PF_MIN,
                                 "min_oos_trades": MIN_OOS_TRADES,
                                 "min_wf_trades": MIN_WF_TRADES,
                                 "degrade_max": 1 - DEGRADE_MIN},
                       "winners": all_winners,
                       "all_results": all_results},
                      fh, indent=2, default=str)

    mins = (time.time() - t_start) / 60
    log(f"DONE in {mins:.1f} min | tested {len(all_results)} "
        f"family/symbol combos | WINNERS: {len(all_winners)}")
    if not all_winners:
        log("No candidate passed ALL gates. This is the expected honest "
            "outcome most of the time - the gates exist to filter luck. "
            "Next levers: tick data, new families, or accept that the edge "
            "must come from execution/discretion, not these signals.")
    else:
        for wrec in all_winners:
            log(f"WINNER {wrec['family']}/{wrec['symbol']}/{wrec['tf']} "
                f"params={wrec['params']} OOS PF={wrec['oos']['pf']:.2f} "
                f"-> port to MQL5 Strategy Tester next")
    return 0


if __name__ == "__main__":
    sys.exit(main())