"""Persistent FINRIX terminal settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "execution_environment": "modal",
    "account_equity": 1000.0,
    "risk_per_trade_pct": 1.0,
    "micro_lot_floor": 0.01,
    "theme": "dark_green",
}
VALID_ENVIRONMENTS = {"modal", "local"}


def load_config(path: Path) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config.update(loaded)
        except (OSError, json.JSONDecodeError):
            pass
    config["execution_environment"] = (
        config["execution_environment"]
        if config["execution_environment"] in VALID_ENVIRONMENTS
        else DEFAULT_CONFIG["execution_environment"]
    )
    for key in ("account_equity", "risk_per_trade_pct", "micro_lot_floor"):
        try:
            config[key] = max(0.0, float(config[key]))
        except (TypeError, ValueError):
            config[key] = DEFAULT_CONFIG[key]
    return config


def save_config(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    normalized = load_config(path)
    normalized.update(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")
    return normalized
