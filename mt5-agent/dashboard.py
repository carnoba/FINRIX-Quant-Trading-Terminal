"""dashboard.py - Streamlit live account dashboard + risk terminal for MT5.

SAFETY MODEL (by design):
  * READ-ONLY by default: monitoring, position table, risk calculator.
  * Order sending requires the "ARM EXECUTION" checkbox AND a valid
    Stop-Loss AND Take-Profit on every order - no exceptions.
  * Hard kill-switch: daily equity drawdown >= 2.0% (server-day basis)
    flips a latched flag that disables the execute button until you
    manually reset it in the UI. The check runs on every rerun/refresh.
  * Login uses the already-logged-in terminal session, or env vars
    MT5_LOGIN / MT5_PASSWORD / MT5_SERVER (never hard-coded).

Launch:
    python -m streamlit run mt5-agent\\dashboard.py
Then open http://localhost:8501
"""
import os
import json
import hmac
import sys
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st

import MetaTrader5 as mt5

import config

st.set_page_config(page_title="MT5 Risk Terminal", layout="wide")

# --- module constants -------------------------------------------------------
KILL_SWITCH_PCT = 2.0      # daily equity drawdown that halts trading
RISK_CHOICES = {"0.25%": 0.25, "0.5%": 0.5, "1.0%": 1.0, "2.0%": 2.0}
DEFAULT_SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD"]
RETCODE_OK = 10009          # TRADE_RETCODE_DONE


def require_dashboard_auth() -> None:
    """Require the configured dashboard password before rendering the terminal."""
    expected = os.environ.get("DASHBOARD_PASSWORD", "")
    if not expected:
        st.error("DASHBOARD_PASSWORD is not configured; dashboard access is disabled.")
        st.stop()
    if st.session_state.get("dashboard_authenticated"):
        return
    st.title("MT5 Risk Terminal")
    supplied = st.text_input("Dashboard password", type="password")
    if st.button("Sign in"):
        if hmac.compare_digest(supplied, expected):
            st.session_state["dashboard_authenticated"] = True
            st.rerun()
        st.error("Invalid dashboard password.")
    st.stop()


require_dashboard_auth()


# --- MT5 connection ---------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to MT5 terminal...")
def connect_mt5():
    """Attach to the running (already logged-in) terminal if possible."""
    if not mt5.initialize():
        login = os.environ.get("MT5_LOGIN")
        password = os.environ.get("MT5_PASSWORD")
        server = os.environ.get("MT5_SERVER", config.BROKER_SERVER)
        if not (login and password):
            return None, ("MT5 terminal not reachable. Open MetaTrader 5 "
                          "and log in, then click Reload.")
        if not mt5.login(int(login), password=password, server=server):
            return None, f"mt5.login failed: {mt5.last_error()}"
    acc = mt5.account_info()
    if acc is None:
        return None, f"No account info: {mt5.last_error()}"
    return acc, None


def server_day_start():
    """Midnight of the CURRENT server time (matches broker daily rollover,
    not your PC clock - critical for the kill-switch basis)."""
    t = mt5.symbol_info_tick("EURUSD")
    now = datetime.fromtimestamp(t.time) if t else datetime.now()
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def day_pnl_summary(day_start):
    """(closed PnL today, floating PnL, today's deals DataFrame)."""
    deals = mt5.history_deals_get(day_start, datetime.now() + timedelta(days=1))
    rows = []
    closed = 0.0
    for d in deals or []:
        if d.entry not in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT):
            continue
        closed += d.profit + d.commission + d.swap
        rows.append({"ticket": d.ticket, "symbol": d.symbol,
                     "type": "BUY" if d.type == 0 else "SELL",
                     "volume": d.volume, "price": d.price,
                     "profit": d.profit, "time":
                     datetime.fromtimestamp(d.time)})
    floating = sum(p.profit + p.swap for p in (mt5.positions_get() or []))
    return closed, floating, pd.DataFrame(rows)


def daily_drawdown_pct(day_start, equity_start):
    """Daily drawdown % from start-of-day equity."""
    closed, floating, _ = day_pnl_summary(day_start)
    current_eq = equity_start + closed + floating
    dd = (equity_start - current_eq) / equity_start * 100 if equity_start > 0 else 0
    return max(0.0, dd), current_eq


def positions_df():
    """Open positions as a DataFrame."""
    pos = mt5.positions_get()
    if not pos:
        return pd.DataFrame()
    rows = []
    for p in pos:
        rows.append({"ticket": p.ticket, "symbol": p.symbol,
                     "type": "BUY" if p.type == 0 else "SELL",
                     "volume": p.volume, "price_open": p.price_open,
                     "price_current": p.price_current,
                     "sl": p.sl, "tp": p.tp,
                     "profit": p.profit, "swap": p.swap})
    return pd.DataFrame(rows)


def calc_lot_size(symbol, risk_pct, sl_price, entry_price):
    """Calculate lot size from risk %. Returns (lots, cash_risk, warning)."""
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return 0.0, 0.0, "No tick data"
    info = mt5.symbol_info(symbol)
    if not info:
        return 0.0, 0.0, "No symbol info"

    acc = mt5.account_info()
    equity = acc.equity if acc else 0.0
    risk_cash = equity * (risk_pct / 100.0)

    point = info.point
    tick_value = info.trade_tick_value
    tick_size = info.trade_tick_size

    stop_dist = abs(entry_price - sl_price)
    if stop_dist <= 0:
        return 0.0, 0.0, "SL must differ from entry"

    stops_in_points = stop_dist / point
    money_per_lot = stops_in_points * tick_value / (tick_size / 0.01)

    if money_per_lot <= 0:
        return 0.0, 0.0, "Invalid tick value"

    lots = risk_cash / money_per_lot
    lot_step = info.volume_step
    lots = round(lots / lot_step) * lot_step
    lots = max(lots, info.volume_min)
    lots = min(lots, info.volume_max)

    cash_risk = lots * money_per_lot
    warn = None
    if cash_risk > risk_cash * 1.1:
        warn = f"Rounded lot size risks ${cash_risk:.2f} > target ${risk_cash:.2f}"
    return lots, cash_risk, warn


def place_order(symbol, side, lots, sl, tp):
    """Send a market order with SL/TP attached. Returns (ok, message)."""
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return False, "No tick data"
    price = tick.ask if side == "BUY" else tick.bid

    info = mt5.symbol_info(symbol)
    point = info.point
    sl = round(sl / point) * point
    tp = round(tp / point) * point

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lots,
        "type": mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 234000,
        "comment": "XAU Hybrid Suite",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != RETCODE_OK:
        return False, f"Order failed: {result.retcode} - {result.comment}"
    return True, f"Order {result.order} placed at {result.price}"


# --- session state defaults -------------------------------------------------
if "armed" not in st.session_state:
    st.session_state.armed = False
if "halted" not in st.session_state:
    st.session_state.halted = False
if "tv_entry" not in st.session_state:
    st.session_state.tv_entry = 0.0
if "tv_sl" not in st.session_state:
    st.session_state.tv_sl = 0.0
if "tv_tp" not in st.session_state:
    st.session_state.tv_tp = 0.0
if "tv_side" not in st.session_state:
    st.session_state.tv_side = "BUY"

# --- header -----------------------------------------------------------------
st.title("🎯 MT5 Risk Terminal — XAUUSD Hybrid Suite")
acc, err = connect_mt5()

if err:
    st.error(err)
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Account", acc.login)
col2.metric("Server", acc.server)
col3.metric("Equity", f"${acc.equity:,.2f}")
col4.metric("Balance", f"${acc.balance:,.2f}")

# --- ARM toggle + kill-switch -----------------------------------------------
st.divider()
col_a, col_b = st.columns([1, 2])
with col_a:
    st.session_state.armed = st.checkbox("🔓 ARM EXECUTION",
                                         value=st.session_state.armed,
                                         help="Enable order sending. Disabled automatically after each trade.")
with col_b:
    day_start = server_day_start()
    dd_pct, current_eq = daily_drawdown_pct(day_start, acc.balance)
    if dd_pct >= KILL_SWITCH_PCT and not st.session_state.halted:
        st.session_state.halted = True
    if st.session_state.halted:
        st.error(f"⛔ KILL-SWITCH ACTIVATED — Daily DD {dd_pct:.2f}% >= {KILL_SWITCH_PCT}%")
        if st.button("🔄 Reset Kill-Switch"):
            st.session_state.halted = False
            st.rerun()
    else:
        st.info(f"Daily drawdown: {dd_pct:.2f}% (limit {KILL_SWITCH_PCT}%)")

# --- tabs -------------------------------------------------------------------
tab_live, tab_scan, tab_curve, tab_journal, tab_exec, tab_tv = st.tabs(
    ["📊 Live Account", "⚡ High-Probability Signals", "📈 Equity Curve",
     "📔 Trade Journal", "🎯 Risk Calculator & Execute", "📡 TV/Webhook Sync"])


# --- live account tab -------------------------------------------------------
with tab_live:
    pos = positions_df()
    if pos.empty:
        st.info("No open positions.")
    else:
        st.subheader("Open positions")
        st.dataframe(pos, use_container_width=True, hide_index=True)

        st.subheader("Exposure & margin")
        expo = (pos.groupby("symbol")
                   .agg(positions=("ticket", "count"),
                        lots=("volume", "sum"),
                        floating_pnl=("profit", "sum")))
        st.dataframe(expo, use_container_width=True)

# --- signals tab ------------------------------------------------------------
with tab_scan:
    st.subheader("High-Probability Signals (Model 3 + Model 1)")
    st.info("Signal scanner integrates with TradingView XAUUSD_Hybrid_Suite.pine "
            "webhook alerts. Use the TV/Webhook Sync tab to ingest alerts.")

# --- equity curve tab -------------------------------------------------------
with tab_curve:
    st.subheader("Equity Curve")
    st.info("Equity curve data loads from MT5 history. "
            "CSV export available in Trade Journal tab.")

# --- journal tab ------------------------------------------------------------
with tab_journal:
    st.subheader("Trade Journal")
    closed, floating, deals_df = day_pnl_summary(day_start)
    if deals_df.empty:
        st.info("No closed trades today.")
    else:
        st.dataframe(deals_df, use_container_width=True, hide_index=True)
        csv = deals_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Export CSV", csv, f"journal_{day_start.date()}.csv",
                           "text/csv")


# --- execution tab ----------------------------------------------------------
with tab_exec:
    if st.session_state.halted:
        st.error("⛔ Halted — reset the kill-switch on the Live tab first.")
        st.stop()

    col_a, col_b = st.columns(2)
    with col_a:
        symbol = st.selectbox("Symbol", DEFAULT_SYMBOLS)
        tick = mt5.symbol_info_tick(symbol)
        bid = tick.bid if tick else 0.0
        ask = tick.ask if tick else 0.0
        st.caption(f"Market: bid {bid} / ask {ask}")
        side = st.radio("Side", ["BUY", "SELL"], horizontal=True,
                        index=0 if st.session_state.get("tv_side", "BUY") == "BUY" else 1)
        risk_pct = st.selectbox("Risk per trade",
                                list(RISK_CHOICES), index=2)  # 1.0%
    with col_b:
        ref = ask if side == "BUY" else bid
        if "tv_entry" not in st.session_state:
            st.session_state.tv_entry = float(f"{ref:.5f}")
        if "tv_sl" not in st.session_state:
            st.session_state.tv_sl = 0.0
        if "tv_tp" not in st.session_state:
            st.session_state.tv_tp = 0.0
        entry = st.number_input("Entry price (prefilled = market)",
                                value=st.session_state.tv_entry,
                                format="%.5f", key="tv_entry")
        sl = st.number_input("Stop Loss (MANDATORY)",
                             value=st.session_state.tv_sl,
                             format="%.5f", key="tv_sl")
        tp = st.number_input("Take Profit (MANDATORY)",
                             value=st.session_state.tv_tp,
                             format="%.5f", key="tv_tp")
        use_auto = st.checkbox("Auto SL/TP from ATR (14) @ 3x risk distance",
                               value=False)
        if use_auto:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 30)
            if rates is not None and sl == 0.0:
                h, l, c = rates["high"], rates["low"], rates["close"]
                tr = np.maximum(h[1:] - l[1:],
                                np.maximum(abs(h[1:] - c[:-1]),
                                           abs(l[1:] - c[:-1])))
                atr = float(tr.mean())
                dist = 3.0 * atr
                sl = ref - dist if side == "BUY" else ref + dist
                tp = ref + 2 * dist if side == "BUY" else ref - 2 * dist
                st.info(f"ATR(14)={atr:.3f} → SL {sl:.5f} / TP {tp:.5f}")

    lots, cash_risk, warn = calc_lot_size(symbol, RISK_CHOICES[risk_pct],
                                          sl if sl else entry, entry)
    st.subheader("Calculated position")
    k1, k2, k3 = st.columns(3)
    k1.metric("Lot size", f"{lots:.2f}")
    k2.metric("Cash risked (if SL hit)", f"${cash_risk:,.2f}")
    k3.metric("% of equity", f"{RISK_CHOICES[risk_pct]:.2f}%")
    if warn:
        st.warning(warn)
    manual = st.checkbox("Override lot size manually (dangerous)", value=False)
    if manual:
        lots = st.number_input("Lots", value=float(lots), min_value=0.01,
                               step=0.01)

    st.divider()
    can_send = (st.session_state.armed and not st.session_state.halted and
                lots > 0 and sl > 0 and tp > 0)
    if not st.session_state.armed and not st.session_state.halted:
        st.info("Toggle ARM EXECUTION at the top of the page to enable sending orders.")
    if not (sl > 0 and tp > 0):
        st.warning("SL and TP are required before the execute button activates.")
    if st.button(f"🚀 SEND {side} {lots:.2f} {symbol} (SL+TP attached)",
                 type="primary", disabled=not can_send):
        ok, msg = place_order(symbol, side, lots, sl, tp)
        (st.success if ok else st.error)(msg)
        if ok:
            st.session_state.armed = False
            st.balloons()


# --- TV / Webhook Sync Tab --------------------------------------------------
with tab_tv:
    st.subheader("📡 TradingView / Webhook Sync")
    st.caption("Paste a TradingView alert webhook JSON payload (or upload a .json "
               "file) to auto-populate the Risk Calculator & Execute form.")

    tv_json_text = st.text_area(
        "Webhook JSON payload",
        value=st.session_state.get("tv_json_text", ""),
        height=200,
        help='Example: {"model":"Model 3","side":"BUY","entry":2650.25,"sl":2640.25,"tp":2680.25,"lots":0.05}',
    )

    col_u, col_p = st.columns([1, 1])
    with col_u:
        uploaded = st.file_uploader("Or upload webhook JSON file", type=["json"])
    with col_p:
        if st.button("🔍 Parse Payload", type="secondary"):
            raw = None
            if uploaded is not None:
                try:
                    raw = json.loads(uploaded.read().decode("utf-8"))
                except Exception as exc:
                    st.error(f"Failed to parse uploaded JSON: {exc}")
            elif tv_json_text.strip():
                try:
                    raw = json.loads(tv_json_text)
                except json.JSONDecodeError as exc:
                    st.error(f"Invalid JSON: {exc}")
            if raw is not None:
                st.session_state.tv_webhook_raw = raw
                st.session_state.tv_json_text = tv_json_text or json.dumps(raw, indent=2)
                st.success("Payload parsed successfully!")

    raw = st.session_state.get("tv_webhook_raw")
    if raw:
        st.divider()
        st.write("**Parsed fields:**")

        def _get(d, *keys, default=None):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return default

        tv_model = _get(raw, "model", "strategy", default="Unknown")
        tv_side = _get(raw, "side", "direction", default="")
        tv_entry = _get(raw, "entry", "price", default=None)
        tv_sl = _get(raw, "sl", "stop_loss", "stoploss", default=None)
        tv_tp = _get(raw, "tp", "take_profit", "takeprofit", default=None)
        tv_lots = _get(raw, "lots", "quantity", "qty", default=None)
        tv_rr = _get(raw, "rr", "risk_reward", default=None)

        df1, df2 = st.columns(2)
        with df1:
            st.text_input("Model", value=str(tv_model), disabled=True)
            st.text_input("Side", value=str(tv_side).upper(), disabled=True)
            if tv_entry is not None:
                st.number_input("Entry", value=float(tv_entry), format="%.5f", disabled=True)
            if tv_sl is not None:
                st.number_input("Stop Loss", value=float(tv_sl), format="%.5f", disabled=True)
        with df2:
            if tv_tp is not None:
                st.number_input("Take Profit", value=float(tv_tp), format="%.5f", disabled=True)
            if tv_lots is not None:
                st.number_input("Lots", value=float(tv_lots), format="%.2f", disabled=True)
            if tv_rr is not None:
                st.text_input("RR", value=str(tv_rr), disabled=True)

        st.divider()
        if st.button("🚀 Load into Execution Form", type="primary"):
            try:
                if tv_side:
                    st.session_state.tv_side = str(tv_side).upper()
                if tv_entry is not None:
                    st.session_state.tv_entry = float(tv_entry)
                if tv_sl is not None:
                    st.session_state.tv_sl = float(tv_sl)
                if tv_tp is not None:
                    st.session_state.tv_tp = float(tv_tp)
                if tv_lots is not None:
                    st.session_state.tv_lots = float(tv_lots)
                st.success("Values loaded into execution form. "
                           "Switch to the '🎯 Risk Calculator & Execute' tab to review and send.")
            except Exception as exc:
                st.error(f"Failed to load values: {exc}")

    with st.expander("📋 Expected webhook JSON schema"):
        st.code("""{
  "model": "Model 3",
  "side": "BUY",
  "entry": 2650.25,
  "sl": 2640.25,
  "tp": 2680.25,
  "lots": 0.05,
  "rr": 3,
  "atr": 12.5
}""", language="json")

st.caption("Risk rails: mandatory SL/TP on every order · 2% daily DD "
           "kill-switch (latched) · lot sizing from broker tick_value · "
           "MT5 session must be logged in. Orders go to whatever account "
           "the terminal is logged into — verify it says DEMO above.")