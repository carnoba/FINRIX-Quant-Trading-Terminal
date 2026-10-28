"""Limit-order SMC scalper - custom simulator (bar mode + tick mode).

backtesting.py cannot cancel/inspect unfilled LIMIT orders (they are
invisible to the strategy), so we simulate order lifecycle ourselves:

SIGNAL (on bar close, rules replicated from the SMC design):
  liquidity sweep -> fresh FVG refined by OB intersection -> LIMIT at zone
  50%, SL = structural level + ATR buffer, TP = rr x risk, distance gate,
  H1-trend + RSI regime filters, daily risk halt (consecutive-loss /
  day-drawdown), one position at a time.

BAR MODE (M5/M15): per-bar fill/exit resolution. When SL and TP are both
inside one bar, order is ambiguous -> report BOTH pessimistic (SL first)
and optimistic (TP first) bounds. Spread applied as bid/ask shift.

TICK MODE: same signals resolved against the real tick stream (bid/ask
columns) for exact fill realism over the available tick windows.

BE: after a closed bar/tick shows >= +1R unrealized, BE level activates;
exit fills exactly AT the BE level when price returns to it.
"""
import argparse
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from mt5_data import load_ohlcv
from run_backtest import trade_metrics
from strategies import ema, atr_wilder
from strategies_v2 import rsi, swings

sys.stdout.reconfigure(line_buffering=True)

LOT_UNITS = {"XAUUSD": 100.0, "EURUSD": 100_000.0}
SPREAD_RATE = 0.0002          # relative bid-ask used in bar mode (config)


def h1_trend(close_arr, index, ema_len=100):
    """EMA on COMPLETED H1 bars only, shifted 1h -> no look-ahead."""
    df = pd.DataFrame({"close": close_arr}, index=pd.Index(index))
    hourly = df.resample("1h").last().dropna()
    t = hourly["close"].ewm(span=ema_len, adjust=False).mean().shift(1)
    idx = pd.Index(index)
    return t.reindex(t.index.union(idx)).sort_index().ffill().reindex(idx).to_numpy()


@dataclass
class Signal:
    bar: int
    time: object
    direction: str
    limit: float
    sl: float
    tp: float
    zone_lo: float
    zone_hi: float
    risk: float


@dataclass
class TradeRec:
    direction: str
    entry_time: object
    exit_time: object
    entry: float
    exit: float
    units: float
    sl: float
    tp: float
    risk: float
    exit_reason: str
    bars_held: int = 0

    @property
    def pnl(self):
        d = 1 if self.direction == "long" else -1
        return (self.exit - self.entry) * d * self.units

    @property
    def r_multiple(self):
        return (self.exit - self.entry) / self.risk * (1 if self.direction == "long" else -1)


def generate_signals(df, rr=1.5, swing_w=2, sweep_lb=6, fvg_lb=6,
                     bos_lb=30, use_ob_refine=True, max_dist_atr=1.5,
                     rsi_lo=35.0, rsi_hi=65.0, trend_ema_h1=100,
                     sl_buffer_atr=0.20, atr_len=14):
    """Scan bars for setups (all filters use only data up to the bar).
    Precomputes sweep/FVG arrays once for speed."""
    o = df["Open"].to_numpy()
    h = df["High"].to_numpy()
    l = df["Low"].to_numpy()
    c = df["Close"].to_numpy()
    idx = df.index
    n = len(df)
    atr = atr_wilder(h, l, c, atr_len)
    r = rsi(c, 14)
    trend = h1_trend(c, idx, trend_ema_h1) if trend_ema_h1 else np.full(n, np.inf)
    ph, pl = swings(h, l, swing_w)

    # ---- precompute: swing reference + sweep flags (single pass each) ------
    swing_lo_ref = np.full(n, -1, dtype=int)
    swing_hi_ref = np.full(n, -1, dtype=int)
    last = -1
    for j in range(n):
        if pl[j] and j >= 3:
            last = j
        swing_lo_ref[j] = last
    last = -1
    for j in range(n):
        if ph[j] and j >= 3:
            last = j
        swing_hi_ref[j] = last

    sweep_long = np.zeros(n, dtype=bool)
    sweep_short = np.zeros(n, dtype=bool)
    for j in range(1, n):
        ref = swing_lo_ref[j]
        if ref >= 0 and ref < j and l[j] < l[ref] and c[j] > l[ref]:
            sweep_long[j] = True
        ref = swing_hi_ref[j]
        if ref >= 0 and ref < j and h[j] > h[ref] and c[j] < h[ref]:
            sweep_short[j] = True

    def rolling_any(flags, w):
        if w <= 1:
            return flags.copy()
        cs = np.cumsum(flags)
        out = flags.copy()
        for i in range(1, n):
            s = cs[i - 1] - (cs[i - 1 - w] if i - 1 - w >= 0 else 0)
            out[i] = flags[i] or s > 0
        return out

    sw_long = rolling_any(sweep_long, sweep_lb)
    sw_short = rolling_any(sweep_short, sweep_lb)

    # ---- precompute FVG zones (refined with OB candle) ---------------------
    zone_lo_l = np.full(n, np.nan)
    zone_hi_l = np.full(n, np.nan)
    zone_lo_s = np.full(n, np.nan)
    zone_hi_s = np.full(n, np.nan)
    for k in range(2, n):
        if l[k] > h[k - 2]:                          # bullish FVG at k
            z_lo, z_hi = h[k - 2], l[k]
            if use_ob_refine and c[k - 1] < o[k - 1]:
                rlo, rhi = max(z_lo, l[k - 1]), min(z_hi, h[k - 1])
                if rlo < rhi:
                    z_lo, z_hi = rlo, rhi
            zone_lo_l[k], zone_hi_l[k] = z_lo, z_hi
        if h[k] < l[k - 2]:                          # bearish FVG at k
            z_lo, z_hi = h[k], l[k - 2]
            if use_ob_refine and c[k - 1] > o[k - 1]:
                rlo, rhi = max(z_lo, l[k - 1]), min(z_hi, h[k - 1])
                if rlo < rhi:
                    z_lo, z_hi = rlo, rhi
            zone_lo_s[k], zone_hi_s[k] = z_lo, z_hi

    sigs = []
    warm = max(trend_ema_h1 + 10, 60)
    for i in range(warm, n):
        a = atr[i]
        if np.isnan(a) or not (rsi_lo < r[i] < rsi_hi):
            continue
        close = c[i]
        for d, sw, zlo_arr, zhi_arr in (
                ("long", sw_long, zone_lo_l, zone_hi_l),
                ("short", sw_short, zone_lo_s, zone_hi_s)):
            if not sw[i]:
                continue
            if trend_ema_h1:
                if d == "long" and not (np.isfinite(trend[i]) and close > trend[i]):
                    continue
                if d == "short" and not (np.isfinite(trend[i]) and close < trend[i]):
                    continue
            # newest FVG zone within fvg_lb bars
            z_lo = z_hi = None
            for k in range(i, max(i - fvg_lb, 2) - 1, -1):
                if zlo_arr[k] == zlo_arr[k]:
                    z_lo, z_hi = zlo_arr[k], zhi_arr[k]
                    break
            if z_lo is None:
                continue
            limit = z_lo + 0.5 * (z_hi - z_lo)
            if abs(close - limit) > max_dist_atr * a:
                continue
            if d == "long":
                sl = z_lo - sl_buffer_atr * a
                risk = limit - sl
                if risk > 0:
                    sigs.append(Signal(i, idx[i], d, limit, sl,
                                       limit + rr * risk, z_lo, z_hi, risk))
            else:
                sl = z_hi + sl_buffer_atr * a
                risk = sl - limit
                if risk > 0:
                    sigs.append(Signal(i, idx[i], d, limit, sl,
                                       limit - rr * risk, z_lo, z_hi, risk))
    return sigs



class Simulator:
    """Order-lifecycle simulator with risk halt + breakeven + one-position rule."""

    def __init__(self, start_cash=10_000.0, risk_pct=1.0, rr=1.5,
                 max_wait_bars=12, daily_max_losses=2,
                 daily_loss_limit_pct=2.0, be_trigger_r=1.0,
                 be_offset_atr_mult=0.05, max_leverage=40, atr_arr=None):
        self.equity = start_cash
        self.risk_pct = risk_pct
        self.rr = rr
        self.max_wait_bars = max_wait_bars
        self.daily_max_losses = daily_max_losses
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.be_trigger_r = be_trigger_r
        self.be_offset_atr_mult = be_offset_atr_mult
        self.max_leverage = max_leverage
        self.atr_arr = atr_arr
        self.trades: list = []
        self.cancelled = 0
        self.halted_days: set = set()
        self.halt_events = 0
        self._day = None
        self._day_start_equity = start_cash
        self._loss_streak = 0

    # --- risk ---------------------------------------------------------------
    def _new_day(self, day):
        if day != self._day:
            self._day = day
            self._day_start_equity = self.equity
            self._loss_streak = 0

    def _on_trade_close(self, rec: TradeRec):
        self.equity += rec.pnl
        self.trades.append(rec)
        if rec.pnl < 0:
            self._loss_streak += 1
        else:
            self._loss_streak = 0
        day_dd = 100.0 * (self._day_start_equity - self.equity) / self._day_start_equity
        if (self._loss_streak >= self.daily_max_losses or
                day_dd >= self.daily_loss_limit_pct):
            if self._day not in self.halted_days:
                self.halted_days.add(self._day)
                self.halt_events += 1

    def halted(self, day):
        self._new_day(day)
        return day in self.halted_days

    def size_units(self, risk_price_dist, price):
        if risk_price_dist <= 0 or price <= 0:
            return 0.0
        units = self.equity * self.risk_pct / 100.0 / risk_price_dist
        cap = self.equity * self.max_leverage / price
        return float(max(min(units, cap), 0.0))

    def _be_level(self, sig: Signal, atr_now):
        off = self.be_offset_atr_mult * (atr_now if atr_now else 0)
        return sig.limit + off if sig.direction == "long" else sig.limit - off

    # --- BAR mode -----------------------------------------------------------
    def run_bars(self, df, sigs, mode="pessimistic"):
        """Resolve fills/exits on OHLC bars with relative spread.
        mode controls same-bar SL+TP ambiguity: 'pessimistic' assumes SL
        first, 'optimistic' assumes TP first."""
        h = df["High"].to_numpy()
        l = df["Low"].to_numpy()
        c = df["Close"].to_numpy()
        idx = df.index
        spread_px = SPREAD_RATE * c
        si = 0
        active = None            # dict(sig, fill_bar, be, entry, units)
        for i in range(len(df)):
            day = idx[i].date()
            self._new_day(day)
            if active is not None:
                sig = active["sig"]
                atr_now = self.atr_arr[i] if self.atr_arr is not None else 0.0
                be = self._be_level(sig, atr_now)
                d = sig.direction
                sl_hit = (l[i] <= sig.sl) if d == "long" else (h[i] >= sig.sl)
                tp_hit = (h[i] >= sig.tp) if d == "long" else (l[i] <= sig.tp)
                if active["be"] and ((l[i] <= be) if d == "long" else (h[i] >= be)):
                    self._close(active, be, "BE", idx, i)
                    active = None
                    continue
                if sl_hit or tp_hit:
                    if sl_hit and tp_hit:
                        px = sig.sl if mode == "pessimistic" else sig.tp
                        reason = "SL" if mode == "pessimistic" else "TP"
                    elif sl_hit:
                        px, reason = sig.sl, "SL"
                    else:
                        px, reason = sig.tp, "TP"
                    self._close(active, px, reason, idx, i)
                    active = None
                    continue
                entry = active["entry"]
                r_mult = ((c[i] - entry) / sig.risk if d == "long"
                          else (entry - c[i]) / sig.risk)
                if r_mult >= self.be_trigger_r:
                    active["be"] = True
                continue
            # try to fill / expire fresh signals (one position at a time)
            while si < len(sigs) and sigs[si].bar + self.max_wait_bars < i:
                self.cancelled += 1
                si += 1
            if si >= len(sigs):
                continue
            sig = sigs[si]
            if sig.bar >= i:
                continue
            if self.halted(day):
                si += 1
                continue
            d = sig.direction
            if (l[i] <= sig.limit) if d == "long" else (h[i] >= sig.limit):
                entry = sig.limit + (spread_px[i] if d == "long" else -spread_px[i])
                units = self.size_units(sig.risk, entry)
                if units > 0:
                    active = {"sig": sig, "fill_bar": i, "be": False,
                              "entry": entry, "units": units}
                si += 1
        return self._trades_df()

    def _close(self, active, px, reason, idx, i):
        sig = active["sig"]
        self._on_trade_close(TradeRec(
            sig.direction, idx[active["fill_bar"]], idx[i], active["entry"], px,
            active["units"], sig.sl, sig.tp, sig.risk, reason,
            i - active["fill_bar"]))

    def _trades_df(self):
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.__dict__ | {"PnL": t.pnl, "R": t.r_multiple}
                             for t in self.trades])

    # --- TICK mode ----------------------------------------------------------
    def run_ticks(self, ticks, sigs, bar_seconds):
        """Resolve against real tick stream (columns: bid, ask, indexed time)."""
        times = ticks.index.to_numpy()          # numpy datetime64 (fast compare)
        bid = ticks["bid"].to_numpy()
        ask = ticks["ask"].to_numpy()
        n = len(ticks)
        si = 0
        active = None      # filled trade state
        pending = None     # (sig, placed_time)
        expiry = None
        day_of = lambda tt: np.datetime64(tt, "D")  # noqa: E731 - fast day key
        for k in range(n):
            t, b, a = times[k], bid[k], ask[k]
            # expire pending
            if pending is not None and t > expiry:
                pending = None
                self.cancelled += 1
            # manage active
            if active is not None:
                sig = active["sig"]
                d = sig.direction
                atr_now = self.atr_arr[active["bar_idx"]] \
                    if self.atr_arr is not None else 0.0
                be = self._be_level(sig, atr_now)
                if d == "long":
                    sl_hit, tp_hit, be_hit = b <= sig.sl, b >= sig.tp, b <= be
                    r_mult = (b - active["entry"]) / sig.risk
                else:
                    sl_hit, tp_hit, be_hit = a >= sig.sl, a <= sig.tp, a >= be
                    r_mult = (active["entry"] - a) / sig.risk
                if active["be"] and be_hit:
                    self._new_day(day_of(t))
                    self._close_tick(active, be, "BE", t)
                    active = None
                    continue
                if sl_hit or tp_hit:
                    if sl_hit and tp_hit:
                        px, reason = sig.sl, "SL"
                    elif sl_hit:
                        px, reason = sig.sl, "SL"
                    else:
                        px, reason = sig.tp, "TP"
                    self._new_day(day_of(t))
                    self._close_tick(active, px, reason, t)
                    active = None
                    continue
                if r_mult >= self.be_trigger_r:
                    active["be"] = True
                continue
            # activate pending on touch
            if pending is not None:
                sig, placed = pending
                if (sig.direction == "long" and a <= sig.limit) or \
                        (sig.direction == "short" and b >= sig.limit):
                    entry = a if sig.direction == "long" else b
                    units = self.size_units(sig.risk, entry)
                    if units > 0:
                        beyond_sl = (entry <= sig.sl) if sig.direction == "long" \
                            else (entry >= sig.sl)
                        if beyond_sl:
                            # Price gapped through the SL before the limit
                            # filled. Real behaviour: broker fills at market
                            # (a/b) and the attached stop triggers instantly
                            # at market -> realize the small spread loss now
                            # instead of "exiting" at a price better than
                            # entry (which would book a phantom SL win).
                            px = b if sig.direction == "long" else a
                            self._new_day(day_of(t))
                            self._on_trade_close(TradeRec(
                                sig.direction, t, t, entry, px, units,
                                sig.sl, sig.tp, sig.risk, "SL", 0))
                        else:
                            active = {"sig": sig, "be": False,
                                      "entry": entry, "units": units,
                                      "bar_idx": sig.bar,
                                      "placed": placed}
                    pending = None
                continue
            # queue next signal when its bar has closed
            while si < len(sigs) and sigs[si].time <= t and not pending:
                if self.halted(day_of(t)):
                    si += 1
                    continue
                pending = (sigs[si], t)
                expiry = t + pd.Timedelta(
                    seconds=bar_seconds * self.max_wait_bars)
                si += 1
        return self._trades_df()

    def _close_tick(self, active, px, reason, t):
        sig = active["sig"]
        self._on_trade_close(TradeRec(
            sig.direction, active["placed"], t, active["entry"], px,
            active["units"], sig.sl, sig.tp, sig.risk, reason, 0))


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def report(trades, sim, label, symbol):
    if trades is None or len(trades) == 0:
        print(f"{label:<22} no trades")
        return None
    pnl = trades["PnL"].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() else float("inf")
    eq = config.START_CASH + pnl.cumsum()
    dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)
    mix = trades["exit_reason"].value_counts().to_dict()
    tot = len(trades)
    lots = trades["units"].mean() / LOT_UNITS.get(symbol, 1e5)
    print(f"{label:<22} trades={tot:>4} win%={100*len(wins)/tot:>5.1f} "
          f"PF={pf:>5.2f} expect$={pnl.mean():>7.2f} MaxDD={dd:>6.1f}% "
          f"avgR={trades['R'].mean():>5.2f}")
    print(f"{'':<22} exits: " + " ".join(
        f"{k}={v}({100*v/tot:.0f}%)" for k, v in sorted(mix.items())) +
        f" | cancelled={sim.cancelled} halted_days={len(sim.halted_days)} "
        f"avg_size={lots:.3f} lots")
    return {"trades": tot, "pf": pf, "win": 100 * len(wins) / tot,
            "dd": dd, "expect": pnl.mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD", choices=list(LOT_UNITS))
    ap.add_argument("--tf", default="M15", choices=["M5", "M15"])
    ap.add_argument("--rr", type=float, default=1.5)
    ap.add_argument("--tick", action="store_true",
                    help="also run tick-level validation on the pulled tick file")
    ap.add_argument("--tick-only", action="store_true",
                    help="skip bar sims; only run tick-level validation")
    args = ap.parse_args()
    if args.tick_only:
        args.tick = True

    df, src = load_ohlcv(args.symbol, args.tf)
    print(f"Bars: {src}")
    h = df["High"].to_numpy(); l = df["Low"].to_numpy(); c = df["Close"].to_numpy()
    atr_arr = atr_wilder(h, l, c, 14)
    sigs = generate_signals(df, rr=args.rr)
    print(f"Signals generated: {len(sigs)} | limit@zone50% "
          f"SL=zone+0.2ATR TP={args.rr}R BE@+1R halt=2loss/2%day")

    cut = int(len(df) * config.OOS_SPLIT)
    if not args.tick_only:
        print("\n--- BAR-MODE SIMULATION (pessimistic SL-first bound) ---")
        for name, sl_df, sl_sigs in (
                ("IS", df.iloc[:cut], [s for s in sigs if s.bar < cut]),
                ("OOS", df.iloc[cut:], [Signal(s.bar - cut, s.time, s.direction, s.limit,
                                               s.sl, s.tp, s.zone_lo, s.zone_hi, s.risk)
                                        for s in sigs if s.bar >= cut]),
                ("FULL", df, sigs)):
            sim = Simulator(rr=args.rr, atr_arr=atr_arr)
            tr = sim.run_bars(sl_df, sl_sigs, "pessimistic")
            report(tr, sim, f"{name} pessimistic", args.symbol)
        print("--- optimistic TP-first bound (upper) ---")
        sim = Simulator(rr=args.rr, atr_arr=atr_arr)
        tr = sim.run_bars(df, sigs, "optimistic")
        report(tr, sim, "FULL optimistic", args.symbol)

    if args.tick:
        tpath = os.path.join(str(config.BASE_DIR), "data",
                             f"{args.symbol}_TICKS_10d.csv")
        if not os.path.exists(tpath):
            tpath = os.path.join(str(config.BASE_DIR), "data",
                                 f"{args.symbol}_TICKS_5d.csv")
        if not os.path.exists(tpath):
            print("[WARN] no tick file found"); return 0
        ticks = pd.read_csv(tpath, usecols=["time", "bid", "ask"],
                            parse_dates=["time"], index_col="time")
        ticks = ticks[["bid", "ask"]].astype("float64").dropna()
        t0, t1 = ticks.index[0], ticks.index[-1]
        mask = (df.index >= t0) & (df.index <= t1)
        sub = df.loc[mask]
        if len(sub) < 50:
            print("[WARN] too few bars inside tick window"); return 0
        atr_sub = atr_wilder(sub["High"].to_numpy(), sub["Low"].to_numpy(),
                             sub["Close"].to_numpy(), 14)
        sigs_sub = [s for s in generate_signals(sub, rr=args.rr)]
        bar_seconds = (sub.index[1] - sub.index[0]).total_seconds()
        print(f"\n--- TICK-LEVEL VALIDATION ({len(ticks):,} ticks, "
              f"{t0.date()}..{t1.date()}, {len(sigs_sub)} signals) ---")
        sim = Simulator(rr=args.rr, atr_arr=atr_sub)
        tr = sim.run_ticks(ticks, sigs_sub, bar_seconds)
        report(tr, sim, "TICK actual bid/ask", args.symbol)
    return 0


if __name__ == "__main__":
    sys.exit(main())




