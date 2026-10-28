with open('mt5-agent/dashboard.py','r',encoding='utf-8',errors='ignore') as f:
    content = f.read()
checks = {
    'tab_tv variable': 'tab_tv' in content,
    'TV/Webhook Sync tab label': 'TV/Webhook Sync' in content,
    'json import': 'import json' in content,
    'tv_json_text session state': 'tv_json_text' in content,
    'Parse Payload button': 'Parse Payload' in content,
    'Load into Execution Form button': 'Load into Execution Form' in content,
    'tv_entry session state': 'tv_entry' in content,
    'tv_sl session state': 'tv_sl' in content,
    'tv_tp session state': 'tv_tp' in content,
    'tv_side session state': 'tv_side' in content,
}
for name, result in checks.items():
    print(f'  {name}: {"OK" if result else "MISSING"}')