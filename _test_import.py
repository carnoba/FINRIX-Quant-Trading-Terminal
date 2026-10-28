import sys
sys.path.insert(0, 'mt5-agent')
import backtest_custom_hybrid as m
print("Import OK")
r, w, s = m.main()
print(f"main() OK - {len(r)} result rows, {len(w)} walk-forward, {len(s)} sensitivity")