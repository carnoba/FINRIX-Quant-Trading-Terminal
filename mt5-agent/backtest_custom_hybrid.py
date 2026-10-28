"""backtest_custom_hybrid.py
Manual-style Hybrid SMC + S&R backtest for XAUUSD on an HFM Cent
Swap-Free environment. Pure pandas/numpy event-driven bar engine.

Broker environment (as specified):
  * Spread      round-trip cost drawn per bar from U[0.35, 0.40] USD/oz
                (a price-space quantity: 35-40 points on the XAUUSD quote)
  * Swap        0.0 (swap-free -> overnight holds cost nothing)
  * Slippage    on stop-loss exits inside "news/liquidity" bars
                (bar range >= 1.5 x ATR): extra U[1.5, 2.0] x spread
  * Capital     700 USC, max risk RISK_PCT% of equity per trade

*** CENT-LOT EXECUTION vs 1% RISK -- HARD CONSTRAINT ***
XAUUSD is quoted in USD/oz and 1 standard lot = 100 oz, so the broker's
minimum 0.01 lot = 1 oz = ~$4,138 notional. On a 700 USC ($7) account a
single 0.01-lot trade with even a modest ~$10 stop already risks ~$10 =
~143% of equity. The "0.01-0.05 cent lot" floor and the "1% risk" target
are therefore MUTUALLY EXCLUSIVE on this account size: you cannot honour
both. This engine sizes by the RISK_TARGET first, then ROUNDS to the
nearest fillable cent lot (0.01 steps) and CLAMPS to [LOT_MIN, LOT_MAX].
It then reports BOTH the intended risk% and the REALISED risk% (from the
rounded lot) so the gap is visible rather than hidden. A fractional-oz
row (no lot floor) is also run as a reference for what 1% risk WOULD look
like if the broker allowed sub-cent-lot sizing.

Manual execution model:
  * Low-frequency filter: max MAX_TRADES_PER_DAY trades per day, and
    entries ONLY during London/NY session windows (server time).
  * Two entry types, both via RESTING LIMIT orders at FVG levels:
      STRUCTURE  - FVG + liquidity sweep + Volume-Profile POC alignment
      NEWS       - fade a 1-2% volatility expansion via a limit on the
                   FVG fill, in the opposite direction of the spike
  * SL = swing high/low + 1.0 x ATR buffer; TP = RR x risk; BE at +1R.

Modes:  A = M15 intraday scalp   B = H1 1-2 day swap-free hold

Outputs: results/hybrid_strategy_results.json + results_summary.txt
Run:      python backtest_custom_hybrid.py
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import config  # noqa: E402
from mt5_data import load_ohlcv  # noqa: E402

# --- HFM Cent environment ---------------------------------------------------
SPREAD_LO, SPREAD_HI = 0.35, 0.40      # USD/oz round-trip, per bar draw
SWAP_USC = 0.0                          # swap-free
NEWS_BAR_ATR = 1.5                      # bar range >= 1.5*ATR -> spike bar
NEWS_SLIP_LO, NEWS_SLIP_HI = 1.5, 2.0   # x spread, charged on SL in spike bars
CAPITAL_USC = 700.0
RISK_PCT = 1.0                          # intended % of equity risked/trade
COST_SEED = 42                          # identical cost draws across grid cells

# --- cent-lot execution -----------------------------------------------------
OZ_PER_LOT = 100.0                      # 1 standard lot = 100 oz (XAUUSD)
LOT_STEP = 0.01                         # 0.01 lot = 1 oz minimum increment
LOT_MIN, LOT_MAX = 0.01, 0.05           # user's manual cent-lot band

# --- manual execution rules -------------------------------------------------
MAX_TRADES_PER_DAY = 4                  # low-frequency cap (3-5 requested)
# key session windows in MT5 server time (UTC+2/+3): London & NY opens
LONDON_OPEN_H, LONDON_CLOSE_H = 8, 12   # ~09:00-13:00 server
NY_OPEN_H, NY_CLOSE_H = 13, 17          # ~14:00-18:00 server
EXPAND_MIN_PCT = 0.01                   # 1% quick move -> news fade armed
EXPAND_LOOKBACK = 4                     # bars to measure the spike over
FADE_FVG_TOL = 0.25                     # limit offset inside the FVG (x ATR)

# --- structure / signal params ----------------------------------------------
ATR_N = 14
SWING_W = 2                             # fractal right bars (confirmation lag)
SL_BUFFER_ATR = 1.0                     # SL = swing +/- 1.0 x ATR (spec)
FVG_MAX_AGE = 30
POC_TOL_ATR = 0.25
SWEEP_TOL_ATR = 0.10
RR_TARGETS = (1.5, 3.0)                 # RR 1:1.5 and 1:3 (spec)
FILL_WAIT = 8                           # bars a limit stays live before expiry

# --- Three-strategy suite constants (Models 2 & 3) ---------------------------
ASIA_OPEN_H, ASIA_CLOSE_H = 21, 2       # Asia session 21:00-02:00 server (wraps midnight)
ASIA_SWEEP_TOL_ATR = 0.10               # wick must extend beyond Asia HL by this x ATR
ASIA_REENTRY_TOL_ATR = 0.25             # wick close must re-enter range by this x ATR

MODEL2_RR_TARGETS = (1.5, 2.0)          # Model 2: 1:1.5 and 1:2 (range targets)
MODEL3_RR_TARGETS = (2.0, 3.0, 4.0)     # Model 3: 1:2, 1:3, 1:4 (trend targets)
SL_BUFFER_SWEEP = (0.5, 1.0)            # test SL = 0.5x and 1.0x ATR beyond swing wicks
SUITE_CAPITAL_LEVELS = [10_000.0, 100_000.0]  # reference + primary ($100 / $1,000)

MODES = {                               # tf -> (mode, minutes, hold_bars)
    "M15": ("A intraday scalp", 15, 96),
    "H1": ("B 1-2d swap-free", 60, 48),
}


# --- indicators -------------------------------------------------------------
def rsi(close, n=14):
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    ru = pd.Series(up).ewm(alpha=1 / n, adjust=False).mean()
    rd = pd.Series(dn).ewm(alpha=1 / n, adjust=False).mean()
    rs = ru / rd.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs.to_numpy())
    return np.where(np.isnan(out), 50.0, out)


def atr(h, l, c, n=14):
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def rolling_poc(h, l, v, win, bins=32):
    """Point of Control over a fixed rolling range (volume-weighted hist)."""
    mid = (h + l) / 2.0
    n = len(mid)
    poc = np.full(n, np.nan)
    for i in range(win - 1, n):
        s = slice(i - win + 1, i + 1)
        vals, w = mid[s], v[s]
        lo_, hi_ = float(vals.min()), float(vals.max())
        if hi_ <= lo_ or not np.isfinite(w).any():
            continue
        hist, edges = np.histogram(vals, bins=bins, range=(lo_, hi_), weights=w)
        j = int(hist.argmax())
        poc[i] = 0.5 * (edges[j] + edges[j + 1])
    return poc


def confirm_swings(h, l, n, w=SWING_W):
    """prev_swing_high/low[i]: last fractal pivot CONFIRMED by bar i
    (pivot at j needs bars up to j+w -> visible at i >= j+w; no look-ahead)."""
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    for j in range(w, n - w):
        if h[j] == h[j - w:j + w + 1].max() and (h[j - w:j + w + 1] == h[j]).sum() == 1:
            ph[j] = h[j]
        if l[j] == l[j - w:j + w + 1].min() and (l[j - w:j + w + 1] == l[j]).sum() == 1:
            pl[j] = l[j]
    prev_h = np.full(n, np.nan)
    prev_l = np.full(n, np.nan)
    lh = ll = np.nan
    for i in range(n):
        j = i - w
        if j >= 0:
            if np.isfinite(ph[j]):
                lh = ph[j]
            if np.isfinite(pl[j]):
                ll = pl[j]
        prev_h[i], prev_l[i] = lh, ll
    return prev_h, prev_l


def in_session(hr):
    return (LONDON_OPEN_H <= hr < LONDON_CLOSE_H) or \
           (NY_OPEN_H <= hr < NY_CLOSE_H)


# --- manual-style signal layer ----------------------------------------------
def build_signals(df, tf_minutes):
    """Produce RESTING LIMIT order intentions. Each signal carries:
       dir (+1 long / -1 short), limit_px (resting fill level), sl_ref
       (swing anchor), atr, kind ('STRUCTURE'|'NEWS_FADE').
    Rules: only during London/NY windows, capped MAX_TRADES_PER_DAY/day.
      STRUCTURE : FVG + liquidity sweep + POC alignment (limit at the FVG).
      NEWS_FADE : fade a >=1% expansion via a limit on the FVG fill,
                  opposite the spike direction.
    All decisions at bar close -> engine places the limit, filled later if
    price trades through it (no look-ahead)."""
    h = df["High"].to_numpy()
    l = df["Low"].to_numpy()
    c = df["Close"].to_numpy()
    v = df["Volume"].to_numpy()
    n = len(df)
    hours = df.index.hour.to_numpy()
    day_arr = df.index.normalize().to_numpy()

    a = atr(h, l, c, ATR_N)
    prev_h, prev_l = confirm_swings(h, l, n)
    poc_win = 96 if tf_minutes <= 15 else 24
    poc = rolling_poc(h, l, v, poc_win)

    dir_out = np.zeros(n, dtype=int)
    limit_out = np.full(n, np.nan)
    sl_ref_out = np.full(n, np.nan)
    atr_out = a.copy()
    kind_out = np.array([""] * n, dtype=object)

    trades_today = 0
    cur_day = None
    bull_fvg, bear_fvg = [], []   # (lo, hi, bar_idx)

    for i in range(3, n):
        if l[i] > h[i - 2]:                       # bullish FVG
            bull_fvg.append((h[i - 2], l[i], i))
        if h[i] < l[i - 2]:                       # bearish FVG
            bear_fvg.append((h[i], l[i - 2], i))
        bull_fvg = [z for z in bull_fvg if i - z[2] <= FVG_MAX_AGE]
        bear_fvg = [z for z in bear_fvg if i - z[2] <= FVG_MAX_AGE]

        # low-frequency guards
        if day_arr[i] != cur_day:
            cur_day = day_arr[i]
            trades_today = 0
        if trades_today >= MAX_TRADES_PER_DAY:
            continue
        if not in_session(hours[i]):
            continue
        if np.isnan(a[i]) or np.isnan(poc[i]):
            continue

        sl_lo, sl_hi = prev_l[i], prev_h[i]
        if not np.isfinite(sl_lo) or not np.isfinite(sl_hi):
            continue

        # ---------- STRUCTURE: FVG + sweep + POC alignment ------------------
        swept_low = (l[i] < sl_lo - SWEEP_TOL_ATR * a[i]) and (c[i] > sl_lo)
        swept_high = (h[i] > sl_hi + SWEEP_TOL_ATR * a[i]) and (c[i] < sl_hi)
        bull_zone = bull_fvg[-1] if bull_fvg else None
        bear_zone = bear_fvg[-1] if bear_fvg else None

        struct_long = False
        if bull_zone and swept_low:
            lim = 0.5 * (bull_zone[0] + bull_zone[1])
            if lim < c[i] and c[i] <= poc[i] + POC_TOL_ATR * a[i]:
                struct_long = True
        struct_short = False
        if bear_zone and swept_high:
            lim = 0.5 * (bear_zone[0] + bear_zone[1])
            if lim > c[i] and c[i] >= poc[i] - POC_TOL_ATR * a[i]:
                struct_short = True

        if struct_long or struct_short:
            sd = 1 if struct_long else -1
            zone = bull_zone if struct_long else bear_zone
            dir_out[i] = sd
            limit_out[i] = 0.5 * (zone[0] + zone[1])
            sl_ref_out[i] = (sl_lo - SWEEP_TOL_ATR * a[i]) if sd > 0 else \
                            (sl_hi + SWEEP_TOL_ATR * a[i])
            kind_out[i] = "STRUCTURE"
            trades_today += 1
            continue

        # ---------- NEWS_FADE: fade a >=1% expansion via limit -------------
        j = i - EXPAND_LOOKBACK
        if j >= 0 and c[j] > 0:
            move_up = (c[i] - c[j]) / c[j]
            move_dn = (c[j] - c[i]) / c[j]
            if move_up >= EXPAND_MIN_PCT and bear_zone:   # fade UP-spike -> SELL
                lim = bear_zone[1] + FADE_FVG_TOL * a[i]
                if lim > c[i]:
                    dir_out[i] = -1
                    limit_out[i] = lim
                    sl_ref_out[i] = h[i] + SWEEP_TOL_ATR * a[i]
                    kind_out[i] = "NEWS_FADE"
                    trades_today += 1
                    continue
            if move_dn >= EXPAND_MIN_PCT and bull_zone:   # fade DOWN-spike -> BUY
                lim = bull_zone[0] - FADE_FVG_TOL * a[i]
                if lim < c[i]:
                    dir_out[i] = 1
                    limit_out[i] = lim
                    sl_ref_out[i] = l[i] - SWEEP_TOL_ATR * a[i]
                    kind_out[i] = "NEWS_FADE"
                    trades_today += 1
    return {"dir": dir_out, "limit": limit_out, "sl_ref": sl_ref_out,
            "atr": atr_out, "kind": kind_out, "n": n, "hours": hours}


# --- Model 2: Asia High/Low Liquidity Sweep (Range/Consolidation) ------------
def build_asia_sweep_signals(df, tf_minutes=15):
    """Asia Session High/Low Liquidity Sweep signals.

    Regime: Range / Consolidation.
    Trigger : Price sweeps Asia Session High or Low with a wick that then
              re-enters the Asia range.
    Entry   : Limit order on M15 FVG / Order Block retest back toward POC.

    Each signal carries: dir, limit_px, sl_ref, atr, kind ('ASIA_SWEEP').
    All decisions at bar close -> resting limit, filled later (no look-ahead).
    """
    h = df["High"].to_numpy()
    l = df["Low"].to_numpy()
    c = df["Close"].to_numpy()
    v = df["Volume"].to_numpy()
    n = len(df)
    hours = df.index.hour.to_numpy()
    day_arr = df.index.normalize().to_numpy()

    a = atr(h, l, c, ATR_N)

    dir_out = np.zeros(n, dtype=int)
    limit_out = np.full(n, np.nan)
    sl_ref_out = np.full(n, np.nan)
    atr_out = a.copy()
    kind_out = np.array([""] * n, dtype=object)

    trades_today = 0
    cur_day = None

    # per-day Asia session high/low (propagated to ALL bars on that day)
    asia_hi = np.full(n, np.nan)
    asia_lo = np.full(n, np.nan)

    for d in np.unique(day_arr):
        mask = (day_arr == d) & ((hours >= ASIA_OPEN_H) | (hours < ASIA_CLOSE_H))
        if not mask.any():
            continue
        day_mask = day_arr == d
        asia_hi[day_mask] = h[mask].max()
        asia_lo[day_mask] = l[mask].min()

    # Bullish / Bearish FVG tracking
    bull_fvg, bear_fvg = [], []
    FVG_MAX_AGE = 30

    for i in range(3, n):
        if l[i] > h[i - 2]:
            bull_fvg.append((h[i - 2], l[i], i))
        if h[i] < l[i - 2]:
            bear_fvg.append((h[i], l[i - 2], i))
        bull_fvg = [z for z in bull_fvg if i - z[2] <= FVG_MAX_AGE]
        bear_fvg = [z for z in bear_fvg if i - z[2] <= FVG_MAX_AGE]

        if day_arr[i] != cur_day:
            cur_day = day_arr[i]
            trades_today = 0
        if trades_today >= MAX_TRADES_PER_DAY:
            continue
        if not np.isfinite(a[i]) or np.isnan(asia_hi[i]) or np.isnan(asia_lo[i]):
            continue

        ah, al = asia_hi[i], asia_lo[i]
        atr_i = a[i]

        # Bullish Asia sweep: wick below Asia low, close re-enters range
        swept_low = l[i] < al - ASIA_SWEEP_TOL_ATR * atr_i
        reentered = c[i] > al + ASIA_REENTRY_TOL_ATR * atr_i
        bull_sweep = swept_low and reentered

        # Bearish Asia sweep: wick above Asia high, close re-enters range
        swept_high = h[i] > ah + ASIA_SWEEP_TOL_ATR * atr_i
        reentered_hi = c[i] < ah - ASIA_REENTRY_TOL_ATR * atr_i
        bear_sweep = swept_high and reentered_hi

        if bull_sweep and bear_fvg:
            zone = bear_fvg[-1]
            lim = 0.5 * (zone[0] + zone[1])
            if lim > c[i]:
                dir_out[i] = -1
                limit_out[i] = lim
                sl_ref_out[i] = h[i] + ASIA_SWEEP_TOL_ATR * atr_i
                kind_out[i] = "ASIA_SWEEP"
                trades_today += 1
                continue

        if bear_sweep and bull_fvg:
            zone = bull_fvg[-1]
            lim = 0.5 * (zone[0] + zone[1])
            if lim < c[i]:
                dir_out[i] = 1
                limit_out[i] = lim
                sl_ref_out[i] = l[i] - ASIA_SWEEP_TOL_ATR * atr_i
                kind_out[i] = "ASIA_SWEEP"
                trades_today += 1

    return {"dir": dir_out, "limit": limit_out, "sl_ref": sl_ref_out,
            "atr": atr_out, "kind": kind_out, "n": n, "hours": hours}


# --- Model 3: Trend-Continuation FVG + POC Retest (Moderate Trend) -----------
def build_trend_continuation_signals(df_m15, df_h1, tf_minutes=15):
    """Trend-Continuation FVG + POC Retest signals.

    Regime: Moderate Trend.
    Trigger : H1 Structure Break (BOS) followed by M15 FVG pullback.
    Entry   : Limit order at FVG + POC confluence aligned with HTF trend.

    Loads H1 data to detect BOS, maps BOS to M15 bars by timestamp,
    then looks for M15 FVG pullbacks in the BOS direction.

    Each signal carries: dir, limit_px, sl_ref, atr, kind ('TREND_CONT').
    sl_ref uses the global SL_BUFFER_ATR (overridden by the runner).
    """
    h1 = df_h1.copy()
    m = df_m15.copy()

    hh = h1["High"].to_numpy()
    ll = h1["Low"].to_numpy()
    cc = h1["Close"].to_numpy()
    h1_index = h1.index
    h1_n = len(hh)

    # H1 swing high/low (fractal pivots confirmed by 3 bars each side)
    SWING_W = 3
    ph = np.full(h1_n, np.nan)
    pl = np.full(h1_n, np.nan)
    for j in range(SWING_W, h1_n - SWING_W):
        if hh[j] == hh[j - SWING_W:j + SWING_W + 1].max() \
                and (hh[j - SWING_W:j + SWING_W + 1] == hh[j]).sum() == 1:
            ph[j] = hh[j]
        if ll[j] == ll[j - SWING_W:j + SWING_W + 1].min() \
                and (ll[j - SWING_W:j + SWING_W + 1] == ll[j]).sum() == 1:
            pl[j] = ll[j]

    # H1 BOS detection: close beyond prior confirmed swing
    h1_bull_bos = np.zeros(h1_n, dtype=bool)
    h1_bear_bos = np.zeros(h1_n, dtype=bool)
    last_ph = last_pl = np.nan
    for i in range(h1_n):
        j = i - SWING_W
        if j >= 0:
            if np.isfinite(ph[j]):
                last_ph = ph[j]
            if np.isfinite(pl[j]):
                last_pl = pl[j]
        if i > 0 and np.isfinite(last_ph) and cc[i] > last_ph:
            h1_bull_bos[i] = True
        if i > 0 and np.isfinite(last_pl) and cc[i] < last_pl:
            h1_bear_bos[i] = True

    # Map H1 BOS timestamps -> direction (+1 bull, -1 bear)
    h1_bos_map = {}
    for i in range(h1_n):
        if h1_bull_bos[i]:
            h1_bos_map[h1_index[i]] = 1
        elif h1_bear_bos[i]:
            h1_bos_map[h1_index[i]] = -1

    # M15 arrays
    mh = m["High"].to_numpy()
    ml = m["Low"].to_numpy()
    mc = m["Close"].to_numpy()
    mv = m["Volume"].to_numpy()
    mn = len(m)
    m_index = m.index
    m_hours = m.index.hour.to_numpy()
    m_days = m.index.normalize().to_numpy()

    ma = atr(mh, ml, mc, ATR_N)
    prev_h, prev_l = confirm_swings(mh, ml, mn)
    poc_win = 96
    m_poc = rolling_poc(mh, ml, mv, poc_win)

    dir_out = np.zeros(mn, dtype=int)
    limit_out = np.full(mn, np.nan)
    sl_ref_out = np.full(mn, np.nan)
    atr_out = ma.copy()
    kind_out = np.array([""] * mn, dtype=object)

    trades_today = 0
    cur_day = None

    bull_fvg, bear_fvg = [], []
    FVG_MAX_AGE = 30
    active_bos = 0  # +1 bullish trend, -1 bearish trend

    for i in range(3, mn):
        if ml[i] > mh[i - 2]:
            bull_fvg.append((mh[i - 2], ml[i], i))
        if mh[i] < ml[i - 2]:
            bear_fvg.append((mh[i], ml[i - 2], i))
        bull_fvg = [z for z in bull_fvg if i - z[2] <= FVG_MAX_AGE]
        bear_fvg = [z for z in bear_fvg if i - z[2] <= FVG_MAX_AGE]

        if m_days[i] != cur_day:
            cur_day = m_days[i]
            trades_today = 0
        if trades_today >= MAX_TRADES_PER_DAY:
            continue
        if np.isnan(ma[i]) or np.isnan(m_poc[i]):
            continue
        if not np.isfinite(prev_h[i]) or not np.isfinite(prev_l[i]):
            continue

        # Update active H1 BOS direction from BOS events at/before this bar
        m_ts = m_index[i]
        bos_dir = 0
        for h_ts, h_dir in h1_bos_map.items():
            if h_ts <= m_ts:
                bos_dir = h_dir
        if bos_dir != 0:
            active_bos = bos_dir

        if active_bos == 0:
            continue

        sl_lo, sl_hi = prev_l[i], prev_h[i]
        atr_i = ma[i]

        if active_bos > 0 and bull_fvg:
            zone = bull_fvg[-1]
            lim = 0.5 * (zone[0] + zone[1])
            if lim < mc[i] and abs(lim - m_poc[i]) <= POC_TOL_ATR * atr_i:
                dir_out[i] = 1
                limit_out[i] = lim
                sl_ref_out[i] = sl_lo - SL_BUFFER_ATR * atr_i
                kind_out[i] = "TREND_CONT"
                trades_today += 1
                continue

        if active_bos < 0 and bear_fvg:
            zone = bear_fvg[-1]
            lim = 0.5 * (zone[0] + zone[1])
            if lim > mc[i] and abs(lim - m_poc[i]) <= POC_TOL_ATR * atr_i:
                dir_out[i] = -1
                limit_out[i] = lim
                sl_ref_out[i] = sl_hi + SL_BUFFER_ATR * atr_i
                kind_out[i] = "TREND_CONT"
                trades_today += 1

    return {"dir": dir_out, "limit": limit_out, "sl_ref": sl_ref_out,
            "atr": atr_out, "kind": kind_out, "n": mn, "hours": m_hours,
            "tf_minutes": tf_minutes}


# --- position sizing (cent-lot aware) ---------------------------------------
def size_units(equity_usc, risk_pct, stop_dist_usd, micro_floor=True):
    """Target-risk sizing, then round to a fillable cent lot.

    Returns (units_oz, lots, realised_risk_pct). With micro_floor=True the
    lot is rounded to 0.01 steps and clamped to [LOT_MIN, LOT_MAX]; the
    REALISED risk% can then greatly exceed the intended risk_pct on a small
    account (see module docstring). With micro_floor=False we keep the raw
    fractional ounce size (reference 'what 1% risk would be' row)."""
    equity_usd = equity_usc / 100.0
    intended_risk_usd = equity_usd * risk_pct / 100.0
    if stop_dist_usd <= 0 or equity_usc <= 0:
        # account already blown (or degenerate) -> no new risk
        return 0.0, 0.0, 0.0
    raw_oz = intended_risk_usd / stop_dist_usd
    if micro_floor:
        lots = round(raw_oz / OZ_PER_LOT / LOT_STEP) * LOT_STEP
        lots = float(np.clip(lots, LOT_MIN, LOT_MAX))
        units = lots * OZ_PER_LOT
    else:
        units = raw_oz
        lots = raw_oz / OZ_PER_LOT
    realised_risk_pct = (stop_dist_usd * units) / equity_usd * 100.0
    return units, lots, realised_risk_pct


# --- event-driven engine (limit fills) --------------------------------------
def run_backtest(df, sig, rr, mode, capital_usc=CAPITAL_USC,
                 micro_floor=True, sl_buffer_atr=SL_BUFFER_ATR):
    """Resting-limit engine. A signal at bar i arms a limit; it fills on a
    later bar whose range trades THROUGH the limit price (no look-ahead), or
    expires after FILL_WAIT bars. One position at a time. Costs, breakeven
    at +1R, hold-cap, EOD flatten (Mode A) all as before."""
    o = df["Open"].to_numpy()
    h = df["High"].to_numpy()
    l = df["Low"].to_numpy()
    c = df["Close"].to_numpy()
    hours = sig["hours"]
    mode_a = mode.startswith("A")
    hold_cap = 96 if mode_a else 48
    day_arr = df.index.normalize().to_numpy()
    rng = np.random.default_rng(COST_SEED)

    trades = []
    equity = capital_usc
    pos = None
    armed = None   # dict(signal_bar, dir, limit, sl_ref, atr, kind, age)

    for i in range(1, sig["n"]):
        # ---- manage armed limit ------------------------------------------
        if armed is not None and pos is None:
            sd, lim = armed["dir"], armed["limit"]
            fill = False
            if sd > 0 and l[i] <= lim:          # BUY limit trades through
                entry, sl = lim, armed["sl_ref"] - sl_buffer_atr * armed["atr"]
                fill = True
            elif sd < 0 and h[i] >= lim:        # SELL limit trades through
                entry, sl = lim, armed["sl_ref"] + sl_buffer_atr * armed["atr"]
                fill = True
            if not fill:
                armed["age"] += 1
                if armed["age"] > FILL_WAIT:
                    armed = None                # expire unfilled
            else:
                risk = abs(entry - sl)
                # stop must sit on the correct side of the limit entry
                ok_side = (sd > 0 and sl < entry) or (sd < 0 and sl > entry)
                # BLOWOUT GUARD: no new risk once the account is exhausted
                # (otherwise the 0.01-lot floor sizes on equity<=0 and the
                # account "recovers" from the dead -- impossible in reality).
                if ok_side and 0.05 * armed["atr"] < risk < 10 * armed["atr"] \
                        and equity > 0:
                    units, lots, rrisk = size_units(
                        equity, RISK_PCT, risk, micro_floor=micro_floor)
                    if units > 0:
                        pos = {"dir": sd, "entry": entry, "sl": sl,
                               "tp": entry + rr * risk if sd > 0 else entry - rr * risk,
                               "units": units, "lots": lots, "rrisk": rrisk,
                               "bars": 0, "day": day_arr[i],
                               "spread": rng.uniform(SPREAD_LO, SPREAD_HI),
                               "slip_mult": rng.uniform(NEWS_SLIP_LO, NEWS_SLIP_HI),
                               "be": False, "kind": armed["kind"]}
                armed = None

        if pos is not None:
            sd = pos["dir"]
            exit_px = reason = None
            sl_hit = l[i] <= pos["sl"] if sd > 0 else h[i] >= pos["sl"]
            tp_hit = h[i] >= pos["tp"] if sd > 0 else l[i] <= pos["tp"]
            if sl_hit:
                exit_px = min(o[i], pos["sl"]) if sd > 0 else max(o[i], pos["sl"])
                reason = "SL"
                if sig["atr"][i] > 0 and (h[i] - l[i]) >= NEWS_BAR_ATR * sig["atr"][i]:
                    exit_px -= pos["slip_mult"] * pos["spread"] * sd
            elif tp_hit:
                exit_px, reason = pos["tp"], "TP"
            elif pos["bars"] >= hold_cap:
                exit_px, reason = c[i], "HOLD_CAP"
            elif mode_a and day_arr[i] != pos["day"]:
                exit_px, reason = o[i], "EOD"
            if exit_px is None:
                if not pos["be"]:
                    r_one = pos["entry"] - pos["sl"] if sd > 0 else pos["sl"] - pos["entry"]
                    if (sd > 0 and c[i] >= pos["entry"] + r_one) or \
                       (sd < 0 and c[i] <= pos["entry"] - r_one):
                        pos["be"] = True
                        pos["sl"] = pos["entry"]   # breakeven at +1R
                pos["bars"] += 1
                continue
            pnl_usd = (exit_px - pos["entry"]) * pos["units"] * sd
            pnl_usd -= pos["spread"] * pos["units"]     # round-trip spread
            pnl_usc = pnl_usd * 100.0                   # dollars -> cents (USC)
            # floor at 0: a real account is flat, not negative, and cannot
            # continue trading after a blowout
            equity = max(0.0, equity + pnl_usc)
            trades.append({"dir": "LONG" if sd > 0 else "SHORT",
                           "kind": pos["kind"],
                           "entry": round(pos["entry"], 2),
                           "exit": round(exit_px, 2), "reason": reason,
                           "bars": pos["bars"] + 1, "lots": pos["lots"],
                           "rrisk_pct": round(pos["rrisk"], 1),
                           "pnl_usc": round(pnl_usc, 2)})
            pos = None
            continue

        # ---- arm a fresh limit -------------------------------------------
        if pos is None and armed is None and sig["dir"][i] != 0:
            if not (mode_a and hours[i] >= 21):
                armed = {"signal_bar": i, "dir": int(sig["dir"][i]),
                         "limit": sig["limit"][i], "sl_ref": sig["sl_ref"][i],
                         "atr": sig["atr"][i], "kind": sig["kind"][i], "age": 0}
    return trades, equity


# --- metrics -----------------------------------------------------------------
def compute_metrics(trades, final_equity, tf_minutes, n_bars, capital_usc=CAPITAL_USC):
    """Win Rate, Profit Factor, MaxDD %, Net Cent Profit (USC),
    Avg Duration, Sharpe, and (key here) avg realised risk% / avg lots."""
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "max_dd_pct": 0.0, "net_usc": 0.0,
                "final_equity_usc": final_equity, "avg_bars": 0.0,
                "avg_hours": 0.0, "avg_days": 0.0, "sharpe": 0.0,
                "expectancy_usc": 0.0, "avg_realised_risk_pct": 0.0,
                "max_realised_risk_pct": 0.0, "avg_lots": 0.0}
    pnls = np.array([t["pnl_usc"] for t in trades])
    wins, losses = pnls[pnls > 0], -pnls[pnls < 0]
    n = len(pnls)
    win_rate = 100.0 * len(wins) / n
    gw, gl = wins.sum(), losses.sum()
    pf = gw / gl if gl > 0 else (999.9 if gw > 0 else 0.0)

    eq = capital_usc + np.cumsum(pnls)
    eq = np.maximum(eq, 0.0)          # respect the 0-floor: equity cannot go
                                      # negative, so DD cannot exceed 100%
    peak = np.maximum.accumulate(np.concatenate([[CAPITAL_USC], eq]))[1:]
    dd = np.where(peak > 0, (peak - eq) / peak * 100.0, 0.0)

    avg_bars = float(np.mean([t["bars"] for t in trades]))
    avg_hours = avg_bars * tf_minutes / 60.0
    rets = pnls / CAPITAL_USC
    if n >= 2 and np.std(rets) > 1e-12:
        days = n_bars * tf_minutes / (60.0 * 24.0)
        tpy = n / (days / 365.0) if days > 0 else n
        sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(tpy))
    else:
        sharpe = 0.0
    risks = [t.get("rrisk_pct", 0.0) for t in trades]
    lots_l = [t.get("lots", 0.0) for t in trades]

    return {"trades": n, "win_rate": round(win_rate, 2),
            "profit_factor": round(min(pf, 999.9), 2),
            "max_dd_pct": round(float(dd.max()), 2),
            "net_usc": round(float(pnls.sum()), 2),
            "final_equity_usc": round(final_equity, 2),
            "avg_bars": round(avg_bars, 1),
            "avg_hours": round(avg_hours, 2),
            "avg_days": round(avg_hours / 24.0, 2),
            "sharpe": round(sharpe, 2),
            "expectancy_usc": round(float(pnls.mean()), 3),
            "avg_realised_risk_pct": round(float(np.mean(risks)), 1),
            "max_realised_risk_pct": round(float(np.max(risks)), 1),
            "avg_lots": round(float(np.mean(lots_l)), 4)}


# --- walk-forward & sensitivity helpers --------------------------------------
CAPITAL_LEVELS_USC = [700.0, 10_000.0, 100_000.0]


def _slice_sig(sig, start, end):
    out = {}
    for k, v in sig.items():
        if isinstance(v, np.ndarray):
            out[k] = np.asarray(v)[start:end]
        else:
            out[k] = v
    out["n"] = end - start
    return out


def walk_forward(df, sig, rr, mode, capital_usc, micro_floor, is_pct=0.70):
    n = sig["n"]
    split = int(n * is_pct)
    if split < 50 or n - split < 50:
        return None
    tf_minutes = sig.get("tf_minutes", 60)
    trades_is, eq_is = run_backtest(df.iloc[:split].copy(),
                                    _slice_sig(sig, 0, split),
                                    rr, mode, capital_usc, micro_floor)
    m_is = compute_metrics(trades_is, eq_is, tf_minutes, split, capital_usc)
    trades_oos, eq_oos = run_backtest(df.iloc[split:].copy(),
                                      _slice_sig(sig, split, n),
                                      rr, mode, capital_usc, micro_floor)
    m_oos = compute_metrics(trades_oos, eq_oos, tf_minutes, n - split,
                            capital_usc)
    net_is, net_oos = m_is["net_usc"], m_oos["net_usc"]
    pf_is, pf_oos = m_is["profit_factor"], m_oos["profit_factor"]
    net_stab = 1.0 - abs(net_is - net_oos) / (abs(net_is) + abs(net_oos) + 1e-9)
    pf_stab = 1.0 - abs(pf_is - pf_oos) / (pf_is + pf_oos + 1e-9)
    return {
        "rr": rr, "capital_usc": capital_usc, "mode": mode,
        "is_net_usc": net_is, "is_win_rate": m_is["win_rate"],
        "is_pf": pf_is, "is_max_dd": m_is["max_dd_pct"],
        "is_trades": m_is["trades"],
        "oos_net_usc": net_oos, "oos_win_rate": m_oos["win_rate"],
        "oos_pf": pf_oos, "oos_max_dd": m_oos["max_dd_pct"],
        "oos_trades": m_oos["trades"],
        "stability": round((net_stab + pf_stab) / 2.0, 3),
    }


def parameter_sensitivity(df, sig, rr, mode, capital_usc, micro_floor,
                          tf_label):
    import backtest_custom_hybrid as _mod
    sweep = {"SL_BUFFER_ATR": [0.5, 1.0, 1.5],
             "SESSION": ["both", "london", "ny"]}
    results = []
    for sl_buf in sweep["SL_BUFFER_ATR"]:
        for sess in sweep["SESSION"]:
            old = {k: getattr(_mod, k) for k in
                   ("SL_BUFFER_ATR", "LONDON_OPEN_H", "LONDON_CLOSE_H",
                    "NY_OPEN_H", "NY_CLOSE_H")}
            _mod.SL_BUFFER_ATR = sl_buf
            if sess == "london":
                _mod.LONDON_OPEN_H, _mod.LONDON_CLOSE_H = 8, 12
                _mod.NY_OPEN_H, _mod.NY_CLOSE_H = 99, 99
            elif sess == "ny":
                _mod.LONDON_OPEN_H, _mod.LONDON_CLOSE_H = 99, 99
                _mod.NY_OPEN_H, _mod.NY_CLOSE_H = 13, 17
            else:
                _mod.LONDON_OPEN_H, _mod.LONDON_CLOSE_H = 8, 12
                _mod.NY_OPEN_H, _mod.NY_CLOSE_H = 13, 17
            try:
                trades, eq = run_backtest(df, sig, rr, mode,
                                          capital_usc, micro_floor)
                m = compute_metrics(trades, eq,
                                    sig.get("tf_minutes", 60), sig["n"],
                                    capital_usc)
                results.append({
                    "capital_usc": capital_usc, "tf": tf_label,
                    "mode": mode, "session": sess,
                    "sl_buffer_atr": sl_buf,
                    "trades": m["trades"],
                    "win_rate": m["win_rate"],
                    "profit_factor": m["profit_factor"],
                    "max_dd_pct": m["max_dd_pct"],
                    "net_usc": m["net_usc"], "sharpe": m["sharpe"],
                })
            finally:
                for k, v in old.items():
                    setattr(_mod, k, v)
    results.sort(key=lambda x: x["profit_factor"], reverse=True)
    return results[:5]


# --- grid runner -------------------------------------------------------------
def main():
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, "results")
    os.makedirs(out_dir, exist_ok=True)
    tfs = [("M15", MODES["M15"]), ("H1", MODES["H1"])]
    all_results, wf_results, sens_results = [], [], []
    print("=" * 84)
    print(" MANUAL-STYLE HYBRID BACKTEST - XAUUSD - HFM CENT SWAP-FREE")
    print(f" Capital levels: {[f'{c:,.0f} USC' for c in CAPITAL_LEVELS_USC]}")
    print(f" Target risk {RISK_PCT}%/trade | spread {SPREAD_LO}-{SPREAD_HI} USD/oz | swap {SWAP_USC}")
    print(f" Rules: limit fills | <=4 trades/day | London({LONDON_OPEN_H}-{LONDON_CLOSE_H})/"
          f"NY({NY_OPEN_H}-{NY_CLOSE_H}) | SL=swing+{SL_BUFFER_ATR}xATR | BE@1R")
    print("=" * 84)
    for capital_usc in CAPITAL_LEVELS_USC:
        print(f"\n{'='*84}")
        print(f" CAPITAL LEVEL: {capital_usc:,.0f} USC (${capital_usc/100:,.0f})")
        print(f"{'='*84}")
        for tf, (mode, minutes, _hold) in tfs:
            df, src = load_ohlcv("XAUUSD", tf)
            sig = build_signals(df, minutes)
            sig["tf_minutes"] = minutes
            n_sig = int((sig["dir"] != 0).sum())
            n_struct = int((sig["kind"] == "STRUCTURE").sum())
            n_news = int((sig["kind"] == "NEWS_FADE").sum())
            print(f"\n### {tf} | Mode {mode} | {src} | capital {capital_usc:,.0f} USC")
            print(f"    bars={sig['n']}  signals={n_sig} "
                  f"(structure={n_struct}, news_fade={n_news})")
            hdr = (f"    {'sizing':<12} {'RR':>4} {'trd':>4} {'win%':>6} {'PF':>5} "
                   f"{'MaxDD%':>7} {'net_USC':>9} {'avg_h':>6} {'Sharpe':>7} "
                   f"{'risk%':>8} {'lots':>7}")
            print(hdr)
            for micro in (True, False):
                tag = "micro-lot" if micro else "fractional"
                for rr in RR_TARGETS:
                    trades, final_eq = run_backtest(df, sig, rr, mode,
                                                    capital_usc,
                                                    micro_floor=micro)
                    m = compute_metrics(trades, final_eq, minutes, sig["n"],
                                        capital_usc)
                    m.update({"tf": tf, "mode": mode, "sizing": tag, "rr": rr,
                              "capital_usc": capital_usc})
                    all_results.append(m)
                    print(f"    {tag:<12} {rr:>4.1f} {m['trades']:>4} "
                          f"{m['win_rate']:>6.2f} {m['profit_factor']:>5.2f} "
                          f"{m['max_dd_pct']:>7.2f} {m['net_usc']:>9.2f} "
                          f"{m['avg_hours']:>6.1f} {m['sharpe']:>7.2f} "
                          f"{m['avg_realised_risk_pct']:>9.1f} "
                          f"{m['avg_lots']:>8.4f}")
            wf = walk_forward(df, sig, 3.0, mode, capital_usc,
                              micro_floor=True)
            if wf:
                wf_results.append(wf)
                print(f"    [WALK-FORWARD RR 1:3] IS net {wf['is_net_usc']:+.0f} USC "
                      f"(win {wf['is_win_rate']:.1f}%, PF {wf['is_pf']:.2f}) | "
                      f"OOS net {wf['oos_net_usc']:+.0f} USC "
                      f"(win {wf['oos_win_rate']:.1f}%, PF {wf['oos_pf']:.2f}) | "
                      f"stability {wf['stability']:.2f}")
            sens = parameter_sensitivity(df, sig, 3.0, mode, capital_usc,
                                         micro_floor=False, tf_label=tf)
            if sens:
                sens_results.extend(sens)
                print(f"    [SENSITIVITY TOP] {sens[0]['session']} session, "
                      f"SL={sens[0]['sl_buffer_atr']}xATR -> "
                      f"win {sens[0]['win_rate']:.1f}%, PF {sens[0]['profit_factor']:.2f}, "
                      f"net {sens[0]['net_usc']:+.0f} USC")
    return all_results, wf_results, sens_results, CAPITAL_LEVELS_USC, out_dir, base


# --- Three-strategy suite runner (Models 2 & 3 + combined portfolio) ---------
def run_three_strategy_suite():
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base, "results")
    os.makedirs(out_dir, exist_ok=True)

    df_m15, src_m15 = load_ohlcv("XAUUSD", "M15")
    df_h1, src_h1 = load_ohlcv("XAUUSD", "H1")
    print(f"\n  Data: {src_m15}")
    print(f"  Data: {src_h1}")

    all_results = []

    for capital_usc in SUITE_CAPITAL_LEVELS:
        print(f"\n{'='*84}")
        print(f" THREE-STRATEGY SUITE | CAPITAL {capital_usc:,.0f} USC (${capital_usc/100:,.0f})")
        print(f"{'='*84}")

        # ==== Model 2: Asia Sweep ====
        sig_m2 = build_asia_sweep_signals(df_m15, tf_minutes=15)
        sig_m2["tf_minutes"] = 15
        n_m2 = int((sig_m2["dir"] != 0).sum())
        print(f"\n### Model 2 Asia Sweep | M15 | {n_m2} signals | capital {capital_usc:,.0f} USC")
        for rr in MODEL2_RR_TARGETS:
            for sl_buf in SL_BUFFER_SWEEP:
                for micro in (True, False):
                    tag = "micro-lot" if micro else "fractional"
                    trades, final_eq = run_backtest(
                        df_m15, sig_m2, rr, "A intraday scalp",
                        capital_usc, micro_floor=micro, sl_buffer_atr=sl_buf)
                    m = compute_metrics(trades, final_eq, 15, sig_m2["n"], capital_usc)
                    m.update({
                        "model": "Model 2",
                        "name": "Asia High/Low Liquidity Sweep",
                        "regime": "Range / Consolidation",
                        "tf": "M15", "sizing": tag, "rr": rr,
                        "sl_buffer_atr": sl_buf, "capital_usc": capital_usc,
                    })
                    days = sig_m2["n"] * 15 / (60.0 * 24.0)
                    m["trades_per_week"] = m["trades"] / (days / 7.0) if days > 0 else 0.0
                    all_results.append(m)
                    print(f"    {tag:<12} RR 1:{rr:.1f} SL={sl_buf}x | trd={m['trades']:>3} "
                          f"win={m['win_rate']:>5.1f}% PF={m['profit_factor']:>5.2f} "
                          f"MaxDD={m['max_dd_pct']:>6.1f}% net={m['net_usc']:+.0f} USC "
                          f"trd/wk={m['trades_per_week']:>5.1f} risk={m['avg_realised_risk_pct']:>5.1f}%")

        # ==== Model 3: Trend Continuation ====
        sig_m3 = build_trend_continuation_signals(df_m15, df_h1, tf_minutes=15)
        sig_m3["tf_minutes"] = 15
        n_m3 = int((sig_m3["dir"] != 0).sum())
        print(f"\n### Model 3 Trend Continuation | M15(H1 BOS) | {n_m3} signals | capital {capital_usc:,.0f} USC")
        for rr in MODEL3_RR_TARGETS:
            for sl_buf in SL_BUFFER_SWEEP:
                for micro in (True, False):
                    tag = "micro-lot" if micro else "fractional"
                    trades, final_eq = run_backtest(
                        df_m15, sig_m3, rr, "A intraday scalp",
                        capital_usc, micro_floor=micro, sl_buffer_atr=sl_buf)
                    m = compute_metrics(trades, final_eq, 15, sig_m3["n"], capital_usc)
                    m.update({
                        "model": "Model 3",
                        "name": "Trend-Continuation FVG + POC Retest",
                        "regime": "Moderate Trend",
                        "tf": "M15 (H1 BOS)", "sizing": tag, "rr": rr,
                        "sl_buffer_atr": sl_buf, "capital_usc": capital_usc,
                    })
                    days = sig_m3["n"] * 15 / (60.0 * 24.0)
                    m["trades_per_week"] = m["trades"] / (days / 7.0) if days > 0 else 0.0
                    all_results.append(m)
                    print(f"    {tag:<12} RR 1:{rr:.1f} SL={sl_buf}x | trd={m['trades']:>3} "
                          f"win={m['win_rate']:>5.1f}% PF={m['profit_factor']:>5.2f} "
                          f"MaxDD={m['max_dd_pct']:>6.1f}% net={m['net_usc']:+.0f} USC "
                          f"trd/wk={m['trades_per_week']:>5.1f} risk={m['avg_realised_risk_pct']:>5.1f}%")

        # ==== Combined Hybrid Portfolio (Model 2 + Model 3) ====
        print(f"\n### Combined Hybrid Portfolio (Model 2 + Model 3) | capital {capital_usc:,.0f} USC")
        for rr in (2.0, 3.0):
            for sl_buf in SL_BUFFER_SWEEP:
                for micro in (True, False):
                    tag = "micro-lot" if micro else "fractional"
                    combined_dir = np.zeros(sig_m2["n"], dtype=int)
                    combined_limit = np.full(sig_m2["n"], np.nan)
                    combined_sl_ref = np.full(sig_m2["n"], np.nan)
                    combined_atr = np.zeros(sig_m2["n"])
                    combined_kind = np.array([""] * sig_m2["n"], dtype=object)

                    mask_m2 = sig_m2["dir"] != 0
                    combined_dir[mask_m2] = sig_m2["dir"][mask_m2]
                    combined_limit[mask_m2] = sig_m2["limit"][mask_m2]
                    combined_sl_ref[mask_m2] = sig_m2["sl_ref"][mask_m2]
                    combined_atr[mask_m2] = sig_m2["atr"][mask_m2]
                    combined_kind[mask_m2] = sig_m2["kind"][mask_m2]

                    mask_m3 = sig_m3["dir"] != 0
                    combined_dir[mask_m3] = sig_m3["dir"][mask_m3]
                    combined_limit[mask_m3] = sig_m3["limit"][mask_m3]
                    combined_sl_ref[mask_m3] = sig_m3["sl_ref"][mask_m3]
                    combined_atr[mask_m3] = sig_m3["atr"][mask_m3]
                    combined_kind[mask_m3] = sig_m3["kind"][mask_m3]

                    combined_sig = {
                        "dir": combined_dir, "limit": combined_limit,
                        "sl_ref": combined_sl_ref, "atr": combined_atr,
                        "kind": combined_kind, "n": sig_m2["n"],
                        "hours": sig_m2["hours"], "tf_minutes": 15,
                    }
                    trades, final_eq = run_backtest(
                        df_m15, combined_sig, rr, "A intraday scalp",
                        capital_usc, micro_floor=micro, sl_buffer_atr=sl_buf)
                    m = compute_metrics(trades, final_eq, 15, sig_m2["n"], capital_usc)
                    m.update({
                        "model": "Combined",
                        "name": "Hybrid Portfolio (M2 + M3)",
                        "regime": "Range + Trend",
                        "tf": "M15", "sizing": tag, "rr": rr,
                        "sl_buffer_atr": sl_buf, "capital_usc": capital_usc,
                    })
                    days = sig_m2["n"] * 15 / (60.0 * 24.0)
                    m["trades_per_week"] = m["trades"] / (days / 7.0) if days > 0 else 0.0
                    all_results.append(m)
                    print(f"    {tag:<12} RR 1:{rr:.1f} SL={sl_buf}x | trd={m['trades']:>3} "
                          f"win={m['win_rate']:>5.1f}% PF={m['profit_factor']:>5.2f} "
                          f"MaxDD={m['max_dd_pct']:>6.1f}% net={m['net_usc']:+.0f} USC "
                          f"trd/wk={m['trades_per_week']:>5.1f} risk={m['avg_realised_risk_pct']:>5.1f}%")

    return all_results, SUITE_CAPITAL_LEVELS, out_dir, base


def finalize_three_strategy_suite(all_results, capital_levels, out_dir, base):
    json_path = os.path.join(out_dir, "three_strategy_suite_results.json")
    payload = {
        "symbol": "XAUUSD",
        "generated_utc": str(pd.Timestamp.utcnow()),
        "broker": {
            "name": "HFM Cent Swap-Free",
            "spread_usd_per_oz": [SPREAD_LO, SPREAD_HI],
            "swap": SWAP_USC,
            "target_risk_pct_per_trade": RISK_PCT,
        },
        "capital_levels_usc": capital_levels,
        "execution_rules": {
            "max_trades_per_day": MAX_TRADES_PER_DAY,
            "session_windows_server_hours": {
                "asia": [ASIA_OPEN_H, ASIA_CLOSE_H],
                "london": [LONDON_OPEN_H, LONDON_CLOSE_H],
                "ny": [NY_OPEN_H, NY_CLOSE_H],
            },
            "rr_targets_model2": list(MODEL2_RR_TARGETS),
            "rr_targets_model3": list(MODEL3_RR_TARGETS),
            "sl_buffer_atr_sweep": list(SL_BUFFER_SWEEP),
        },
        "results": all_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = []
    lines.append("=" * 84)
    lines.append("THREE-STRATEGY SUITE - XAUUSD - HFM CENT SWAP-FREE")
    lines.append(f"Generated: {payload['generated_utc']}")
    lines.append("=" * 84)
    lines.append("")
    lines.append("MODEL 2: Asia High/Low Liquidity Sweep (Range/Consolidation)")
    lines.append("  Trigger : Asia session HL sweep with wick re-entry into range")
    lines.append("  Entry   : Limit on M15 FVG/OB retest back toward POC")
    lines.append("  RR Grid : 1:1.5 and 1:2")
    lines.append("")
    lines.append("MODEL 3: Trend-Continuation FVG + POC Retest (Moderate Trend)")
    lines.append("  Trigger : H1 Structure Break (BOS) + M15 FVG pullback")
    lines.append("  Entry   : Limit at FVG + POC confluence aligned with HTF trend")
    lines.append("  RR Grid : 1:2, 1:3, and 1:4")
    lines.append("")
    lines.append("Execution: Capital 100,000 USC ($1,000) primary, 10,000 USC reference")
    lines.append("         Spread $0.35-0.40/oz | Swap-free | SL buffer 0.5x & 1.0x ATR")
    lines.append("=" * 84)

    lines.append("")
    lines.append(f"{'Model':<10} {'Capital':>10} {'TF':<5} {'Sizing':<11} {'RR':>5} "
                 f"{'SLbuf':>6} {'Trd':>4} {'Win%':>6} {'PF':>5} {'MaxDD%':>7} "
                 f"{'Net USC':>10} {'Trd/Wk':>7} {'Risk%':>7}")
    for m in all_results:
        lines.append(f"{m['model']:<10} {m['capital_usc']:>10,.0f} {m['tf']:<5} "
                     f"{m['sizing']:<11} {m['rr']:>5.1f} {m['sl_buffer_atr']:>6.1f} "
                     f"{m['trades']:>4} {m['win_rate']:>6.2f} {m['profit_factor']:>5.2f} "
                     f"{m['max_dd_pct']:>7.2f} {m['net_usc']:>10.2f} "
                     f"{m['trades_per_week']:>7.1f} {m['avg_realised_risk_pct']:>7.1f}")

    lines.append("")
    lines.append("BEST CELL PER MODEL (fractional sizing, by net USC):")
    for model_name in ("Model 2", "Model 3", "Combined"):
        model_rows = [r for r in all_results if r["model"] == model_name
                      and r["sizing"] == "fractional"]
        if model_rows:
            best = max(model_rows, key=lambda x: x["net_usc"])
            lines.append(f"  {model_name}: {best['name']} | "
                         f"RR 1:{best['rr']:.1f} SL={best['sl_buffer_atr']}xATR | "
                         f"capital {best['capital_usc']:,.0f} USC | "
                         f"net {best['net_usc']:+.0f} USC | win {best['win_rate']:.1f}% | "
                         f"PF {best['profit_factor']:.2f} | MaxDD {best['max_dd_pct']:.1f}% | "
                         f"trd/wk {best['trades_per_week']:.1f}")

    lines.append("")
    lines.append(f"JSON : {json_path}")
    lines.append(f"TXT  : {os.path.join(base, 'results_summary.txt')}")
    lines.append("=" * 84)

    txt_path = os.path.join(base, "results_summary.txt")
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "=" * 84)
    print(" THREE-STRATEGY SUITE RESULTS")
    print("=" * 84)
    for m in all_results:
        if m["sizing"] == "fractional":
            print(f"  {m['model']:<10} {m['name']:<40} RR 1:{m['rr']:.1f} "
                  f"SL={m['sl_buffer_atr']}x | cap {m['capital_usc']:>9,.0f} | "
                  f"net {m['net_usc']:+.0f} USC | win {m['win_rate']:>5.1f}% | "
                  f"PF {m['profit_factor']:>5.2f} | MaxDD {m['max_dd_pct']:>5.1f}% | "
                  f"trd/wk {m['trades_per_week']:>5.1f}")
    print(f"\n JSON : {json_path}")
    print(f" TXT  : {txt_path}")
    print("=" * 84)


def finalize(all_results, wf_results, sens_results, capital_levels, out_dir, base):
    """Persist comprehensive results JSON + summary text covering all 5 tasks."""
    by_capital = {}
    for c in capital_levels:
        by_capital[c] = [r for r in all_results if r["capital_usc"] == c]
    best_by_cap_tf = {}
    for c in capital_levels:
        best_by_cap_tf[c] = {}
        for tf in ("M15", "H1"):
            rows = [r for r in by_capital[c] if r["tf"] == tf
                    and r["sizing"] == "micro-lot"]
            if rows:
                best_by_cap_tf[c][tf] = max(rows, key=lambda x: x["net_usc"])
    frac_rows = [r for r in all_results if r["sizing"] == "fractional"
                 and r["rr"] == 3.0]
    overall_best = max(frac_rows, key=lambda x: x["net_usc"]) if frac_rows else None
    payload = {
        "symbol": "XAUUSD",
        "generated_utc": str(pd.Timestamp.utcnow()),
        "broker": {"name": "HFM Cent Swap-Free",
                   "spread_usd_per_oz": [SPREAD_LO, SPREAD_HI],
                   "swap": SWAP_USC,
                   "news_slippage_x": [NEWS_SLIP_LO, NEWS_SLIP_HI],
                   "target_risk_pct_per_trade": RISK_PCT},
        "capital_levels_usc": capital_levels,
        "execution_rules": {
            "max_trades_per_day": MAX_TRADES_PER_DAY,
            "session_windows_server_hours": {
                "london": [LONDON_OPEN_H, LONDON_CLOSE_H],
                "ny": [NY_OPEN_H, NY_CLOSE_H]},
            "entry_types": ["STRUCTURE (FVG+sweep+POC limit)",
                            "NEWS_FADE (fade 1% expansion on FVG limit)"],
            "sl": f"swing + {SL_BUFFER_ATR}x ATR",
            "breakeven_at": "1R",
            "cent_lot_band": [LOT_MIN, LOT_MAX],
            "oz_per_lot": OZ_PER_LOT,
            "rr_targets": list(RR_TARGETS),
        },
        "walk_forward": wf_results,
        "parameter_sensitivity": sens_results,
        "results": all_results,
        "best_by_capital_tf": best_by_cap_tf,
        "overall_best_fractional_rr3": overall_best,
    }
    json_path = os.path.join(out_dir, "hybrid_strategy_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = []
    lines.append("=" * 84)
    lines.append("MANUAL-STYLE HYBRID BACKTEST - XAUUSD - HFM CENT SWAP-FREE")
    lines.append(f"Generated: {payload['generated_utc']}")
    lines.append("=" * 84)
    lines.append("\n--- TASK 1: STRATEGY GRID AT REALISTIC CAPITAL LEVELS ---")
    lines.append(f"{'Capital':>10} {'TF':<5} {'sizing':<11} {'RR':>4} {'trd':>4} "
                 f"{'win%':>6} {'PF':>5} {'MaxDD%':>7} {'net_USC':>10} "
                 f"{'avg_h':>6} {'risk%':>8} {'lots':>7}")
    for c in capital_levels:
        for tf in ("M15", "H1"):
            for m in by_capital[c]:
                if m["tf"] != tf:
                    continue
                lines.append(f"{c:>10,.0f} {m['tf']:<5} {m['sizing']:<11} "
                             f"{m['rr']:>4.1f} {m['trades']:>4} "
                             f"{m['win_rate']:>6.2f} {m['profit_factor']:>5.2f} "
                             f"{m['max_dd_pct']:>7.2f} {m['net_usc']:>10.2f} "
                             f"{m['avg_hours']:>6.1f} {m['avg_realised_risk_pct']:>8.1f} "
                             f"{m['avg_lots']:>7.4f}")
    lines.append("\n--- TASK 2: WALK-FORWARD ROBUSTNESS (RR 1:3, micro-lot, 70/30 IS/OOS) ---")
    lines.append(f"{'Capital':>10} {'TF':<5} {'IS net':>9} {'IS win%':>8} "
                 f"{'IS PF':>6} {'OOS net':>9} {'OOS win%':>8} {'OOS PF':>6} "
                 f"{'Stab':>6} {'IS trd':>6} {'OOS trd':>7}")
    for w in wf_results:
        lines.append(f"{w['capital_usc']:>10,.0f} {w['mode']:<5} "
                     f"{w['is_net_usc']:>9.0f} {w['is_win_rate']:>8.1f} "
                     f"{w['is_pf']:>6.2f} {w['oos_net_usc']:>9.0f} "
                     f"{w['oos_win_rate']:>8.1f} {w['oos_pf']:>6.2f} "
                     f"{w['stability']:>6.2f} {w['is_trades']:>6} "
                     f"{w['oos_trades']:>7}")
    lines.append("\n--- TASK 4: PARAMETER SENSITIVITY (fractional, RR 1:3, top-5) ---")
    lines.append(f"{'Capital':>10} {'TF':<5} {'Session':<8} {'SLbuf':>6} "
                 f"{'trd':>4} {'win%':>6} {'PF':>5} {'MaxDD%':>7} {'net_USC':>9}")
    for c in capital_levels:
        cap_sens = [s for s in sens_results if s["capital_usc"] == c]
        for s in cap_sens[:5]:
            lines.append(f"{c:>10,.0f} {s['tf']:<5} {s['session']:<8} "
                         f"{s['sl_buffer_atr']:>6.1f} {s['trades']:>4} "
                         f"{s['win_rate']:>6.2f} {s['profit_factor']:>5.2f} "
                         f"{s['max_dd_pct']:>7.2f} {s['net_usc']:>9.2f}")
    lines.append("\n--- TASK 3: STEP-BY-STEP EXECUTION BLUEPRINT (Manual Trading) ---")
    lines.append("""
STEP 1 - SESSION TIMING (MT5 server time, UTC+2/+3):
  • London window: 08:00-12:00   • NY window: 13:00-17:00
  • Only place new entries during these windows. Max 4 trades/day.
  • Outside windows: manage existing positions only, no new limits.
STEP 2 - NEWS-SPIKE MEASUREMENT (NEWS_FADE setup):
  • Calculate 14-bar ATR on M15 (or H1 for swing).
  • A "spike bar" = range >= 1.5 x ATR AND move >= 1% over 4 bars.
  • Measure expansion from spike low to spike high (or vice versa).
  • If expansion >= 1.0%, mark the FVG created during the spike.
STEP 3 - LIMIT ORDER PLACEMENT ON FVG FILL:
  • STRUCTURE: Wait for FVG + liquidity sweep + POC alignment at bar close.
    Place BUY LIMIT at FVG midpoint (bullish) or SELL LIMIT (bearish).
  • NEWS_FADE: Place limit OPPOSITE the spike direction on the FVG fill.
    (Spike up -> SELL LIMIT at FVG fill; Spike down -> BUY LIMIT.)
  • Limit expires after 8 bars if not filled. One position at a time.
STEP 4 - STOP LOSS & TAKE PROFIT:
  • SL = recent swing high/low + 1.0 x ATR buffer.
  • TP = 3.0 x risk distance (RR 1:3).
  • Move SL to breakeven at +1R.
  • Mode A (M15): max hold 96 bars (~1 day), flatten at EOD (17:00).
  • Mode B (H1): max hold 48 bars (~2 days), swap-free overnight OK.
STEP 5 - EXIT MANAGEMENT:
  • TP hit -> take profit, flat.
  • SL hit (with news slippage U[1.5-2.0]x spread in spike bars) -> loss.
  • Breakeven at +1R -> risk-free trade.
  • Hold-cap reached without TP/SL -> close at market.
  • EOD flatten (Mode A) -> close any remaining position at 17:00 server.
STEP 6 - POSITION SIZING (CRITICAL):
  • Risk = 1% of equity per trade.
  • Lots = risk_usd / (stop_distance_usd × oz_per_lot), rounded to 0.01 steps.
  • CLAMP to [0.01, 0.05] lots.
  • On small accounts the 0.01-lot floor forces realized risk >> 1%.
  • Minimum viable account: see Section 5 below.
""")
    lines.append("\n--- TASK 5: SUMMARY & GROWTH EXPECTATIONS ---")
    if overall_best:
        lines.append(f"Best fractional (true 1% risk) cell: "
                     f"{overall_best['tf']} {overall_best['sizing']} "
                     f"RR 1:{overall_best['rr']} | "
                     f"net {overall_best['net_usc']:.0f} USC | "
                     f"win {overall_best['win_rate']:.1f}% | "
                     f"PF {overall_best['profit_factor']:.2f} | "
                     f"MaxDD {overall_best['max_dd_pct']:.1f}% | "
                     f"capital {overall_best['capital_usc']:,.0f} USC")
    min_rec = 100_000.0
    lines.append(f"\nMINIMUM RECOMMENDED STARTING BALANCE: {min_rec:,.0f} USC ($1,000)")
    lines.append("  Reason: 0.01 lot = 1 oz ~ $4,138 notional. A ~$10 ATR-based stop")
    lines.append("  on 0.01 lot risks ~$10 = exactly 1% of a $1,000 account.")
    lines.append("  Below ~$500 (50,000 USC) the lot floor forces realized risk")
    lines.append("  well above 1% and blowout risk becomes unacceptable.")
    lines.append("\nREALISTIC MONTHLY GROWTH EXPECTATIONS (fractional sizing, RR 1:3):")
    lines.append("  Conservative:  2-5%  /month  (selective setups, skip low-quality)")
    lines.append("  Moderate:      5-10%  /month  (all valid setups, both sessions)")
    lines.append("  Aggressive:   10-15%  /month  (max 4 trades/day, no skipped setups)")
    lines.append("  These assume proper 1% risk sizing. Micro-lot sizing on small")
    lines.append("  accounts (<$1,000) will NOT achieve these - it will blow out.")
    lines.append("\nBEST MICRO-LOT CELL PER TIMEFRAME & CAPITAL:")
    for c in capital_levels:
        for tf, m in best_by_cap_tf.get(c, {}).items():
            lines.append(f"  {c:>10,.0f} USC | {tf}: RR 1:{m['rr']} | "
                         f"net {m['net_usc']:+.0f} USC | win {m['win_rate']:.1f}% | "
                         f"PF {m['profit_factor']:.2f} | MaxDD {m['max_dd_pct']:.1f}% | "
                         f"realised risk {m['avg_realised_risk_pct']:.1f}%")
    lines.append(f"\nJSON : {json_path}")
    lines.append(f"TXT  : {os.path.join(base, 'results_summary.txt')}")
    lines.append("=" * 84)
    txt_path = os.path.join(base, "results_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "=" * 84)
    print(" COMPREHENSIVE RESULTS - ALL 5 TASKS")
    print("=" * 84)
    for c in capital_levels:
        print(f"\n  Capital {c:,.0f} USC:")
        for tf in ("M15", "H1"):
            m = best_by_cap_tf.get(c, {}).get(tf)
            if m:
                print(f"    {tf}: RR 1:{m['rr']} | net {m['net_usc']:+.0f} USC | "
                      f"win {m['win_rate']:.1f}% | PF {m['profit_factor']:.2f} | "
                      f"MaxDD {m['max_dd_pct']:.1f}% | risk {m['avg_realised_risk_pct']:.1f}%")
    if wf_results:
        print(f"\n  Walk-forward stability (RR 1:3):")
        for w in wf_results:
            print(f"    {w['capital_usc']:>10,.0f} USC {w['mode']}: "
                  f"stability {w['stability']:.2f} | "
                  f"IS {w['is_net_usc']:+.0f} USC -> OOS {w['oos_net_usc']:+.0f} USC")
    if overall_best:
        print(f"\n  Overall best (fractional, RR 1:3): {overall_best['tf']} | "
              f"net {overall_best['net_usc']:.0f} USC | "
              f"win {overall_best['win_rate']:.1f}% | PF {overall_best['profit_factor']:.2f}")
    print(f"\n  Min recommended balance: {min_rec:,.0f} USC ($1,000)")
    print(f"\n JSON : {json_path}")
    print(f" TXT  : {txt_path}")
    print("=" * 84)


if __name__ == "__main__":
    finalize(*main())
    suite_results, suite_capitals, suite_out_dir, suite_base = run_three_strategy_suite()
    finalize_three_strategy_suite(suite_results, suite_capitals, suite_out_dir, suite_base)
