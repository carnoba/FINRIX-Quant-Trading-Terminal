"""Environment check: locate MT5 terminal, connect, and report account/symbol info.
Zero execution risk: read-only, no orders, no credentials.
"""
import sys
import MetaTrader5 as mt5

print("Python:", sys.version)
print("MT5 package version:", mt5.version())

ok = mt5.initialize()
print("initialize():", ok, "| last_error:", mt5.last_error())
if not ok:
    print("Terminal NOT found/started. Possible reasons:")
    print("  - MT5 not installed on this PC")
    print("  - portable install -> pass path= to mt5.initialize()")
    sys.exit(1)

term = mt5.terminal_info()
print("Terminal path:", term.path)
print("Terminal connected:", term.connected)

acc = mt5.account_info()
if acc is None:
    print("No account logged in. Open MT5 terminal and log in manually, then re-run.")
else:
    print(f"Account: login={acc.login} server={acc.server} name={acc.name}")
    print(f"  balance={acc.balance} equity={acc.equity} leverage=1:{acc.leverage} currency={acc.currency}")

for sym in ("EURUSD", "XAUUSD", "GBPUSD"):
    info = mt5.symbol_info(sym)
    if info is not None:
        mt5.symbol_select(sym, True)
        print(f"{sym}: visible={info.visible} point={info.point} digits={info.digits} "
              f"trade_tick_size={info.trade_tick_size} spread_cur={(info.ask - info.bid) if info.ask else 'n/a'}")
    else:
        print(f"{sym}: NOT FOUND on this broker")

# test historical data availability (read-only)
rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 10)
print("EURUSD H1 last 10 bars:", "OK" if rates is not None else f"FAILED {mt5.last_error()}")

mt5.shutdown()
print("DONE")
