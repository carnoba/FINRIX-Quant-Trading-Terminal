"""Baseline strategies for Step-1 validation (backtesting.py 0.6.x).

Both strategies:
  - risk a fixed % of equity per trade (position size from stop distance)
  - ALWAYS set a hard stop-loss
  - one position at a time (no pyramiding)
  - signal evaluated on bar close; fill happens on the NEXT bar open
    (backtesting.py default -> no same-bar / look-ahead fills)

STRATEGY 1 - EMACrossATR: classic trend-following baseline.
STRATEGY 2 - OrderBlockRetest: simplified Smart-Money-Concepts baseline
             (bearish candle = order block, entry on retest of the zone,
              EMA-200 trend filter, 2R target).
NOTE: honest research baselines, not tuned "holy grails".
"""
import numpy as np
import pandas as pd
from backtesting import Strategy

import config as config_module


# --------------------------------------------------------------------------
# indicator helpers (no TA-Lib dependency)
# --------------------------------------------------------------------------
def ema(values, length):
    return pd.Series(values).ewm(span=length, adjust=False).mean().to_numpy()


def atr_wilder(high, low, close, length):
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean().to_numpy()


# --------------------------------------------------------------------------
# Strategy 1: EMA crossover + ATR risk management
# --------------------------------------------------------------------------
class EMACrossATR(Strategy):
    fast = 20
    slow = 50
    atr_len = 14
    sl_atr = 1.5     # stop   = 1.5 x ATR
    tp_atr = 3.0     # target = 3.0 x ATR  (2R)
    risk_pct = 1.0   # risk 1% of equity per trade

    def init(self):
        close = self.data.Close
        self.ema_f = self.I(ema, close, self.fast, overlay=True)
        self.ema_s = self.I(ema, close, self.slow, overlay=True)
        self.atr = self.I(atr_wilder, self.data.High, self.data.Low, close,
                          self.atr_len, overlay=False)

    def _position_size(self, stop_dist, entry_price):
        """Risk-based size in WHOLE units (backtesting.py rejects fractional
        units > 1). 1% of equity risked / stop distance = units to trade."""
        if stop_dist <= 0 or np.isnan(stop_dist) or entry_price <= 0:
            return 0
        risk_cash = self.equity * self.risk_pct / 100.0
        units = int(round(risk_cash / stop_dist))
        # margin safety cap (stays below broker leverage limit)
        max_units = int(self.equity * config_module.MAX_LEVERAGE / entry_price)
        return max(min(units, max_units), 0)


    def next(self):
        if len(self.data) < 2:
            return
        prev_diff = self.ema_f[-2] - self.ema_s[-2]
        now_diff = self.ema_f[-1] - self.ema_s[-1]
        crossed_up = prev_diff <= 0 < now_diff
        crossed_dn = prev_diff >= 0 > now_diff
        a = self.atr[-1]
        if np.isnan(a):
            return

        if self.position:
            if self.position.is_long and crossed_dn:
                self.position.close()
            elif self.position.is_short and crossed_up:
                self.position.close()
            return

        entry_ref = self.data.Close[-1]  # fill ref ≈ next bar open
        if crossed_up:
            sl = entry_ref - self.sl_atr * a
            tp = entry_ref + self.tp_atr * a
            size = self._position_size(entry_ref - sl, entry_ref)
            if size:
                try:
                    self.buy(size=size, sl=sl, tp=tp)
                except ValueError:
                    return  # next-open gap made SL/TP invalid vs fill - skip
        elif crossed_dn:
            sl = entry_ref + self.sl_atr * a
            tp = entry_ref - self.tp_atr * a
            size = self._position_size(sl - entry_ref, entry_ref)
            if size:
                try:
                    self.sell(size=size, sl=sl, tp=tp)
                except ValueError:
                    return  # gap-invalidated entry - skip


# --------------------------------------------------------------------------
# Strategy 2: simplified Order Block retest (SMC baseline)
# --------------------------------------------------------------------------
class OrderBlockRetest(Strategy):
    lookback = 12          # how far back to search for a fresh order block
    ob_min_body_atr = 0.5  # displacement candle body must be >= 0.5 ATR
    sl_buffer_atr = 0.25   # stop placed 0.25 ATR beyond the OB zone
    rr = 2.0               # reward:risk target (2R)
    risk_pct = 1.0
    atr_len = 14
    trend_filter = 200     # 0 = disabled; else EMA trend filter length
    max_zone_age = 10      # OB valid for max 10 bars after formation

    def init(self):
        close = self.data.Close
        self.atr = self.I(atr_wilder, self.data.High, self.data.Low, close,
                          self.atr_len, overlay=False)
        if self.trend_filter:
            self.ema_t = self.I(ema, close, self.trend_filter, overlay=True)
        self._used = set()  # bar indices of OBs already consumed

    def _find_ob(self):
        """Return (direction, zone_lo, zone_hi, ob_index) of newest valid OB,
        scanning backwards. Respects trend filter and OB age/usage."""
        hi, lo, op, cl = self.data.High, self.data.Low, self.data.Open, self.data.Close
        a = self.atr[-1]
        if np.isnan(a):
            return None
        start = max(2, len(self.data) - self.lookback)
        for i in range(len(self.data) - 2, start - 1, -1):
            if i in self._used or (len(self.data) - 1 - i) > self.max_zone_age:
                continue
            # bullish OB: bearish candle + bullish displacement above its high
            if (cl[i] < op[i] and cl[i + 1] > hi[i]
                    and (cl[i + 1] - op[i + 1]) >= self.ob_min_body_atr * a):
                if self.trend_filter == 0 or self.data.Close[-1] > self.ema_t[-1]:
                    return ("long", lo[i], hi[i], i)
            # bearish OB: bullish candle + bearish displacement below its low
            if (cl[i] > op[i] and cl[i + 1] < lo[i]
                    and (op[i + 1] - cl[i + 1]) >= self.ob_min_body_atr * a):
                if self.trend_filter == 0 or self.data.Close[-1] < self.ema_t[-1]:
                    return ("short", lo[i], hi[i], i)
        return None

    def _size(self, stop_dist, entry_price):
        """Whole-unit risk-based size with margin safety cap."""
        if stop_dist <= 0 or entry_price <= 0:
            return 0
        risk_cash = self.equity * self.risk_pct / 100.0
        units = int(round(risk_cash / stop_dist))
        max_units = int(self.equity * config_module.MAX_LEVERAGE / entry_price)
        return max(min(units, max_units), 0)


    def next(self):
        if self.position or len(self.data) < self.trend_filter + 5:
            return
        ob = self._find_ob()
        if ob is None:
            return
        direction, zone_lo, zone_hi, ob_idx = ob

        # price must retrace INTO the OB zone on this bar
        if not (self.data.Low[-1] <= zone_hi and self.data.High[-1] >= zone_lo):
            return

        entry = self.data.Close[-1]  # fill ref ≈ next bar open (actual fill may gap)
        a = self.atr[-1]
        if direction == "long":
            sl = zone_lo - self.sl_buffer_atr * a
            dist = entry - sl
            units = self._size(dist, entry)
            if units > 0 and dist > 0:
                self._used.add(ob_idx)
                try:
                    self.buy(size=units, sl=sl, tp=entry + self.rr * dist)
                except ValueError:
                    return  # next-open gap made SL/TP invalid vs fill - skip
        else:
            sl = zone_hi + self.sl_buffer_atr * a
            dist = sl - entry
            units = self._size(dist, entry)
            if units > 0 and dist > 0:
                self._used.add(ob_idx)
                try:
                    self.sell(size=units, sl=sl, tp=entry - self.rr * dist)
                except ValueError:
                    return  # gap-invalidated entry - skip


STRATEGIES = {
    "ema": EMACrossATR,
    "smc": OrderBlockRetest,
}

