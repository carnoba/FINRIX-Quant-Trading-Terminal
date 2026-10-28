"""verify_volbreak_ticks.py - Direction B: tick-level re-check of the ONE
near-miss candidate (volbreak / XAUUSD M15, Donchian k=20, exp=1.0,
sl_atr=3.0, rr=2.5) that the miner rejected at OOS PF=0.74.

Hypotheses tested (increasingly aggressive at removing cost):
  A baseline      : market orders next-open + full config spread (miner)
  B maker entries : LIMIT orders resting at signal close, filled only when
                    price trades through (no spread on entry)
  C raw spread    : actual avg XAUUSD tick spread from our tick files
  D combined      : maker + raw spread + no commission
  E maker @ close-0.5ATR (deeper limit offset)
Bar-mode: pessimistic SL-first. Final section replays the candidate against
the REAL tick stream (exact bid/ask) over the tick-data window.
Verdict: OOS PF > 1.2 with >= 30 OOS trades => graduates to MQL5 tester.
"""
import sys

import numpy as np
import pandas as pd

import config
from mt5_data import load_ohlcv
from strategies import atr_wilder
from strategy_miner import metrics_from_trades, sig_volbreak

PARAMS = {"chan": "donchian", "k": 20, "exp": 1.0, "sl_atr": 3.0, "rr": 2.5}


def engine(df, arrays, i0, i1, spread, maker=False, offset_atr=0.0,
           commission=True):
    """Bar-mode replay of volbreak signals. maker=True: resting limit at
    close +/- offset_atr*ATR, filled when bar trades through; expire 5 bars."""
    d, sl, tp, _exl, _exs, _mh = arrays
    seg = df.iloc[i0:i1]
    o = seg["Open"].to_numpy()
    h = seg["High"].to_numpy()
    l = seg["Low"].to_numpy()
    c = seg["Close"].to_numpy()
    atr = atr_wilder(h, l, c, 14)
    ds, sls, tps = d[i0:i1], sl[i0:i1], tp[i0:i1]
    n = len(seg)
    equity = config.START_CASH
    trades = []
    pos, pending = None, None
    for i in range(n):
        if pending is not None:
            pending = pending[:4] + (pending[4] + 1,)
            if pending[4] > 5:
                pending = None
        if pos is None and pending is None and i >= 1 and ds[i - 1] != 0:
            sd = int(ds[i - 1])
            stop, target = sls[i - 1], tps[i - 1]
            if maker:
                off = offset_atr * atr[i - 1] if not np.isnan(atr[i - 1]) else 0.0
                limit_px = c[i - 1] - off if sd > 0 else c[i - 1] + off
                pending = (sd, limit_px, stop, target, 0)
            else:
                entry = o[i] * (1.0 + spread) if sd > 0 else o[i] * (1.0 - spread)
                risk = (entry - stop) if sd > 0 else (stop - entry)
                if 0 < risk and ((target > entry) if sd > 0 else (target < entry)):
                    units = equity * 0.01 / risk
                    units = min(units, equity * config.MAX_LEVERAGE / entry)
                    if units > 0:
                        pos = {"dir": sd, "entry": entry, "sl": stop,
                               "tp": target, "units": units, "risk": risk}
        if pos is None and pending is not None:
            sd, limit_px, stop, target, _age = pending
            filled = (l[i] <= limit_px) if sd > 0 else (h[i] >= limit_px)
            if filled:
                entry = limit_px          # maker fill - no spread cost
                risk = (entry - stop) if sd > 0 else (stop - entry)
                if 0 < risk and ((target > entry) if sd > 0 else (target < entry)):
                    units = equity * 0.01 / risk
                    units = min(units, equity * config.MAX_LEVERAGE / entry)
                    if units > 0:
                        pos = {"dir": sd, "entry": entry, "sl": stop,
                               "tp": target, "units": units, "risk": risk}
                pending = None
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
            if px is None and i == n - 1:
                px, reason = c[i], "EOF"
            if px is not None:
                pnl = sd * (px - pos["entry"]) * pos["units"]
                if commission:
                    pnl -= config.COMMISSION * pos["entry"] * pos["units"]
                equity += pnl
                trades.append({"pnl": pnl,
                               "r": sd * (px - pos["entry"]) / pos["risk"]})
                pos = None
    return metrics_from_trades(trades)


def main():
    df, src = load_ohlcv("XAUUSD", "M15")
    n = len(df)
    cut = int(n * config.OOS_SPLIT)
    arrays = sig_volbreak(df, PARAMS)
    print(f"Data: {src}")
    print(f"Params: {PARAMS}")

    # measure the real average spread from our tick files
    ticks = pd.read_csv(f"{config.DATA_DIR}/XAUUSD_TICKS_10d.csv",
                        usecols=["bid", "ask"])
    raw_abs = float((ticks["ask"] - ticks["bid"]).mean())
    raw_rel = float(((ticks["ask"] - ticks["bid"]) / ticks["bid"]).mean())
    print(f"[info] measured avg XAUUSD spread: {raw_abs:.3f} price units "
          f"({raw_rel:.6f} relative vs config {config.SPREAD:.5f})\n")

    print(f"{'variant':<34} {'IS PF':>6} {'OOS PF':>7} {'OOSwin':>7} "
          f"{'OOSn':>5}  verdict")
    variants = [
        ("A baseline (miner settings)", config.SPREAD, False, 0.0, True),
        ("B maker limit @ close", config.SPREAD, True, 0.0, True),
        ("C raw tick spread", raw_rel, False, 0.0, True),
        ("D maker + raw spread + 0 comm", raw_rel, True, 0.0, False),
        ("E maker @ close-0.5ATR", config.SPREAD, True, 0.5, True),
    ]
    winner = False
    for name, spread, maker, off, comm in variants:
        is_m = engine(df, arrays, 0, cut, spread, maker, off, comm)
        oos_m = engine(df, arrays, cut, n, spread, maker, off, comm)
        ok = (oos_m["pf"] > 1.2 and oos_m["trades"] >= 30)
        winner = winner or ok
        print(f"{name:<34} {is_m['pf']:>6.2f} {oos_m['pf']:>7.2f} "
              f"{oos_m['win']:>6.1f}% {oos_m['trades']:>5}  "
              f"{'PASS->MQL5' if ok else 'fail'}")

    # --- exact tick-stream replay over the tick window ---------------------
    print("\n--- exact tick-stream replay (real bid/ask) ---")
    tk = pd.read_csv(f"{config.DATA_DIR}/XAUUSD_TICKS_10d.csv",
                     parse_dates=["time"], index_col="time")
    tk = tk[["bid", "ask"]].astype(float).dropna()
    t0, t1 = tk.index[0], tk.index[-1]
    sub = df.loc[(df.index >= t0) & (df.index <= t1)]
    if len(sub) < 50:
        print("[WARN] tick window too small for replay")
        return 0
    sub_arrays = sig_volbreak(sub, PARAMS)
    d, sl, tp = sub_arrays[0], sub_arrays[1], sub_arrays[2]
    bar_sec = (sub.index[1] - sub.index[0]).total_seconds()
    b = tk["bid"].to_numpy()
    a = tk["ask"].to_numpy()
    times = tk.index
    equity = config.START_CASH
    trades = []
    pos = None
    sigs = [(i, int(d[i]), sl[i], tp[i]) for i in range(len(sub)) if d[i] != 0]
    si = 0
    t_start = times[0]
    for k in range(len(tk)):
        t, bb, aa = times[k], b[k], a[k]
        if pos is not None:
            sd, estop, etp, ent, units, risk = pos
            px = reason = None
            if sd > 0:
                if bb <= estop:
                    px, reason = bb, "SL"
                elif bb >= etp:
                    px, reason = etp, "TP"
            else:
                if aa >= estop:
                    px, reason = aa, "SL"
                elif aa <= etp:
                    px, reason = etp, "TP"
            if px is not None:
                pnl = sd * (px - ent) * units
                pnl -= config.COMMISSION * ent * units
                equity += pnl
                trades.append({"pnl": pnl, "r": sd * (px - ent) / risk})
                pos = None
            continue
        # fill earliest unfilled signal whose bar time has passed
        elapsed = (t - t_start).total_seconds()
        while si < len(sigs) and sigs[si][0] * bar_sec <= elapsed:
            bar_i, sd, s_sl, s_tp = sigs[si]
            si += 1
            entry = aa if sd > 0 else bb    # taker: pay the spread
            risk = (entry - s_sl) if sd > 0 else (s_sl - entry)
            if 0 < risk and ((s_tp > entry) if sd > 0 else (s_tp < entry)):
                units = equity * 0.01 / risk
                units = min(units, equity * config.MAX_LEVERAGE / entry)
                if units > 0:
                    pos = (sd, s_sl, s_tp, entry, units, risk)
                    break
    if pos is not None:   # force flat at last tick
        sd, estop, etp, ent, units, risk = pos
        px = b[-1] if sd > 0 else a[-1]
        pnl = sd * (px - ent) * units - config.COMMISSION * ent * units
        equity += pnl
        trades.append({"pnl": pnl, "r": sd * (px - ent) / risk})
    m = metrics_from_trades(trades)
    print(f"tick window {t0.date()}..{t1.date()} | signals={len(sigs)} "
          f"trades={m['trades']} win={m['win']:.1f}% PF={m['pf']:.2f} "
          f"expect=${m['expect']:.2f}")
    print("\nVERDICT RULE: any variant with OOS PF>1.2 & n>=30 graduates.")
    print(f"RESULT: {'at least one variant PASSED' if winner else 'NO variant passed - volbreak stays dead'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())