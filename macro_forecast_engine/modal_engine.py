"""Root-level Modal entrypoint for the macro forecast engine."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_IMPLEMENTATION = (
    Path(__file__).resolve().parents[1]
    / "mt5-agent"
    / "macro_forecast_engine"
    / "modal_engine.py"
)
_SPEC = spec_from_file_location("_macro_forecast_modal_engine", _IMPLEMENTATION)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load Modal engine from {_IMPLEMENTATION}")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

app = _MODULE.app
main = _MODULE.main
