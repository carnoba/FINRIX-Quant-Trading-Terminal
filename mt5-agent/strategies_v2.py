"""SMC v2: upgraded Smart-Money-Concepts baseline with confluence filters.

Improvements over v1 (each is a plain, computable rule - no hindsight):
  1. Break of Structure (BOS): close breaks the most recent confirmed swing
     high (bullish) / swing low (bearish).
  2. Liquidity sweep: wick pierces a recent swing level and closes back
     beyond it (stop-hunt reclaim).
  3. RSI(14) regime filter: no entries in overbought/oversold extremes.
  4. Volatility regime filter: ATR(14) must exceed 0.8 x SMA(ATR,50).
  5. Multi-timeframe trend: daily EMA(200) from COMPLETED daily bars only
     (shifted 1 day) -> no look-ahead on intraday decisions.
  6. Stricter displacement: OB body >= 1.0 x ATR (v1 used 0.5).

Entry model stays an OB-zone retest with hard SL and fixed 2R target.
"""
import numpy as np
import pandas as pd
from backtesting import Strategy

from strategies import ema, atr_wilder


def rsi(close, length=14):
    c = pd.Series(close)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50).to_numpy()


def daily_trend_filter(close_arr, index, ema_len=200):
    """EMA(200) on COMPLETED daily bars, shifted 1 day, mapped to bar index.
    Using only closed daily bars avoids look-ahead on intraday decisions."""
    df = pd.DataFrame({"close": close_arr}, index=pd.Index(index))
    daily = df.resample("1D").last().dropna()
    trend_daily = daily["close"].ewm(span=ema_len, adjust=False).mean().shift(1)
    idx = pd.Index(index)
    aligned = trend_daily.reindex(trend_daily.index.union(idx)).sort_index().ffill()
    return aligned.reindex(idx).to_numpy()


def swings(high, low, w=2):
    """Pivot highs/lows (fractal). Returns boolean arrays."""
    n = len(high)
    ph = np.zeros(n, dtype=bool)
    pl = np.zeros(n, dtype=bool)
    for i in range(w, n - w):
        hh = high[i - w:i + w + 1]
        ll = low[i - w:i + w + 1]
        if high[i] == hh.max() and (hh == high[i]).sum() == 1:
            ph[i] = True
        if low[i] == ll.min() and (ll == low[i]).sum() == 1:
            pl[i] = True
    return ph, pl


class SMCv2(Strategy):
    # structure
    swing_w = 2            # fractal window
    bos_lookback = 30      # structure search window
    sweep_lookback = 5     # liquidity-sweep search window
    # order block
    ob_min_body_atr = 1.0  # displacement must be strong (1.0 x ATR)
    sl_buffer_atr = 0.3
    max_zone_age = 8
    rr = 2.0
    # filters
    rsi_len = 14
    atr_len = 14
    atr_sma_len = 50
    atr_regime_mult = 0.8   # skip quiet markets: ATR <= 0.8 x ATR-SMA
    trend_ema = 200         # daily EMA filter (0 = off)
    # risk
    risk_pct = 1.0

    def init(self):
        close = np.asarray(self.data.Close, dtype=float)
        high = np.asarray(self.data.High, dtype=float)
        low = np.asarray(self.data.Low, dtype=float)

        self.atr = self.I(atr_wilder, self.data.High, self.data.Low,
                          self.data.Close, self.atr_len, overlay=False)
        self.atr_sma = self.I(ema, self.atr, self.atr_sma_len, overlay=False)
        self.rsi = self.I(rsi, self.data.Close, self.rsi_len, overlay=False)
        if self.trend_ema:
            self.trend = self.I(lambda: daily_trend_filter(
                close, self.data.index, self.trend_ema), overlay=False)
        self.ph, self.pl = swings(high, low, self.swing_w)
        self._used = set()

    # --- structure helpers --------------------------------------------------
    def _last_swing(self, arr, upto, min_age=3):
        for j in range(upto - 1, max(upto - self.bos_lookback, 0) - 1, -1):
            if arr[j] and (upto - j) >= min_age:
                return j
        return None

    def _sweep(self, direction):
        """True if a recent bar wicked past a swing level and closed back."""
        n = len(self.data)
        lo, cl, hi = self.data.Low, self.data.Close, self.data.High
        for j in range(n - 1, max(n - 1 - self.sweep_lookback, 1) - 1, -1):
            ref = self._last_swing(self.pl if direction == "long" else self.ph, j)
            if ref is None or ref >= j:
                continue
            if direction == "long" and lo[j] < lo[ref] and cl[j] > lo[ref]:
                return True
            if direction == "short" and hi[j] > hi[ref] and cl[j] < hi[ref]:
                return True
        return False

    def _bos(self, direction):
        n = len(self.data)
        if direction == "long":
            ref = self._last_swing(self.ph, n - 1)
            return ref is not None and self.data.Close[-1] > self.data.High[ref]
        ref = self._last_swing(self.pl, n - 1)
        return ref is not None and self.data.Close[-1] < self.data.Low[ref]

    def _find_ob(self, direction):
        hi, lo, op, cl = self.data.High, self.data.Low, self.data.Open, self.data.Close
        a = self.atr[-1]
        if np.isnan(a):
            return None
        n = len(self.data)
        for i in range(n - 2, max(n - 1 - self.max_zone_age, 1) - 1, -1):
            if i in self._used:
                continue
            body = abs(cl[i + 1] - op[i + 1])
            if direction == "long" and cl[i] < op[i] and cl[i + 1] > hi[i] \
                    and body >= self.ob_min_body_atr * a:
                return (lo[i], hi[i], i)
            if direction == "short" and cl[i] > op[i] and cl[i + 1] < lo[i] \
                    and body >= self.ob_min_body_atr * a:
                return (lo[i], hi[i], i)
        return None

    def _size(self, stop_dist, entry):
        if stop_dist <= 0 or entry <= 0:
            return 0
        from strategies import config_module
        units = int(round(self.equity * self.risk_pct / 100.0 / stop_dist))
        max_units = int(self.equity * config_module.MAX_LEVERAGE / entry)
        return max(min(units, max_units), 0)

    def next(self):
        n = len(self.data)
        if self.position or n < self.trend_ema + 10:
            return
        a = self.atr[-1]
        r = self.rsi[-1]
        if np.isnan(a) or np.isnan(self.atr_sma[-1]) or a <= self.atr_regime_mult * self.atr_sma[-1]:
            return  # volatility regime filter
        if not (30 < r < 70):   # RSI regime filter (both directions)
            return

        for direction in ("long", "short"):
            if self.trend_ema:
                t = self.trend[-1]
                if direction == "long" and not (np.isfinite(t) and self.data.Close[-1] > t):
                    continue
                if direction == "short" and not (np.isfinite(t) and self.data.Close[-1] < t):
                    continue
            if not (self._bos(direction) or self._sweep(direction)):
                continue  # need BOS or liquidity-sweep confluence
            ob = self._find_ob(direction)
            if ob is None:
                continue
            zone_lo, zone_hi, ob_idx = ob
            if not (self.data.Low[-1] <= zone_hi and self.data.High[-1] >= zone_lo):
                continue  # wait for retest
            entry = self.data.Close[-1]
            if direction == "long":
                sl = zone_lo - self.sl_buffer_atr * a
                dist = entry - sl
                units = self._size(dist, entry)
                if units > 0 and dist > 0:
                    self._used.add(ob_idx)
                    try:
                        self.buy(size=units, sl=sl, tp=entry + self.rr * dist)
                    except ValueError:
                        pass
            else:
                sl = zone_hi + self.sl_buffer_atr * a
                dist = sl - entry
                units = self._size(dist, entry)
                if units > 0 and dist > 0:
                    self._used.add(ob_idx)
                    try:
                        self.sell(size=units, sl=sl, tp=entry - self.rr * dist)
                    except ValueError:
                        pass
            return  # one decision per bar


STRATEGIES2 = {
    "smc2": SMCv2,
}

