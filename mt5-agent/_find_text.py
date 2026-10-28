with open('backtest_custom_hybrid.py','r') as f:
    content = f.read()
# Count occurrences of the print line
target = "f\"trd/wk={m['trades_per_week']:>5.1f} risk={m['avg_realised_risk_pct']:>5.1f}%\""
count = content.count(target)
print(f'Occurrences: {count}')
# Find all positions
start = 0
while True:
    pos = content.find(target, start)
    if pos == -1:
        break
    line_num = content[:pos].count('\n') + 1
    print(f'  Line {line_num}: char {pos}')
    start = pos + 1