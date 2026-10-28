"""Login helper for the MT5 demo account (setup only - not needed at runtime
if the terminal is already logged in).

Credentials are read from ENVIRONMENT VARIABLES so nothing sensitive is
hard-coded in any file:
    MT5_LOGIN=114303307
    MT5_PASSWORD=<your password>
    MT5_SERVER=MetaQuotes-Demo   (optional - defaults to config.BROKER_SERVER)

Usage (cmd):
    set MT5_LOGIN=114303307&& set MT5_PASSWORD=xxxx&& python mt5-agent\\setup_login.py

Never commit real credentials; treat passwords like this file treats them.
"""
import os
import sys

import MetaTrader5 as mt5

import config


def main() -> int:
    login = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER", config.BROKER_SERVER)

    if not mt5.initialize():
        code, msg = mt5.last_error()
        print(f"[ERROR] initialize failed: {code} {msg}")
        return 2

    acc = mt5.account_info()
    if acc is not None and str(acc.login) == login:
        print(f"Already logged in: {acc.login}@{acc.server} - no action needed.")
        mt5.shutdown()
        return 0

    if not login or not password:
        print("[ERROR] Set MT5_LOGIN and MT5_PASSWORD environment variables.")
        mt5.shutdown()
        return 2

    ok = mt5.login(login=int(login), password=password, server=server)
    if not ok:
        code, msg = mt5.last_error()
        print(f"[ERROR] login failed: code {code} ({msg})")
        print("Check: server name exact ('%s'), credentials, and that the" % server)
        print("account is not locked; or just log in inside the terminal UI.")
        mt5.shutdown()
        return 1

    acc = mt5.account_info()
    print(f"LOGIN OK: {acc.login}@{acc.server}")
    print(f"  balance={acc.balance} equity={acc.equity} leverage=1:{acc.leverage} {acc.currency}")
    mt5.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
