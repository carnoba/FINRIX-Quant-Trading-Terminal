"""JSON-safe conversion helpers shared by local and Modal execution paths."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None:
        return None
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if type(value).__name__ == "NaTType":
        return None
    return value
