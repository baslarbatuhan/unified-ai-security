"""tools/ — gateway-side tool implementations.

Each tool exposes a single `call(**kwargs) -> Dict[str, Any]` function.
The registry below maps tool names (as registered in
`fusion_gateway/engine._register_gateway_demo_schemas`) to their
handlers, so the runner can `invoke(name, args)` after the gateway's
ParameterValidator has cleared a tool call.

Three tools shipped (Hafta 14):
    weather_forecast  → open-meteo REST API
    stock_quote       → Yahoo Finance public chart endpoint
    calc_evaluate     → simpleeval (AST-restricted, never `eval()`)

Design notes:
  * Tools never raise on bad input — they return `{"error": "..."}`
    so the runner can surface the failure on the CSV row without
    aborting the whole suite.
  * Network tools use `httpx` (already a transitive dep via FastAPI)
    with explicit timeouts so a hung upstream can't stall a run.
  * `calc_evaluate` uses `simpleeval` which only permits arithmetic
    expressions over numbers — `__import__`, attribute access, and
    function calls are all rejected at parse time. The gateway's
    allow-list regex (`^[0-9+\\-*/().\\s]+$`) is the primary screen;
    simpleeval is defence-in-depth.
"""
from __future__ import annotations

from typing import Any, Callable, Dict


class ToolNotFoundError(KeyError):
    """Raised when invoke() is called with an unknown tool name."""


# Registry populated by each tool module via _register() below. Kept in
# this central module so callers do `from tools import invoke` without
# caring which sub-module owns which name.
_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {}


def _register(name: str, handler: Callable[..., Dict[str, Any]]) -> None:
    """Add a handler to the registry. Re-registration is allowed (last
    write wins) so tests can swap in fakes without touching the source."""
    _REGISTRY[name] = handler


def available_tools() -> Dict[str, Callable[..., Dict[str, Any]]]:
    """Return a copy of the registry — callers can introspect without
    mutating the live dict."""
    return dict(_REGISTRY)


def invoke(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Run a registered tool. Returns the tool's payload dict on success,
    or `{"error": "<message>"}` when the tool itself raises. Unknown tool
    names raise `ToolNotFoundError` (caller decides how to surface that —
    typically as a 'tool_not_registered' violation on the CSV row).
    """
    handler = _REGISTRY.get(tool_name)
    if handler is None:
        raise ToolNotFoundError(f"tool {tool_name!r} is not registered")
    try:
        # Tools accept kwargs by spec; passing the args dict via ** keeps
        # call sites uniform.
        return handler(**(args or {}))
    except Exception as exc:  # noqa: BLE001 — surface any tool-level failure
        return {"error": f"{type(exc).__name__}: {exc}"}


# Import each tool module so they self-register on import. Done at the
# bottom so the registry is empty above and the _register helper has
# already been defined.
from tools import calculator as _calculator_mod  # noqa: E402,F401
from tools import stock as _stock_mod            # noqa: E402,F401
from tools import weather as _weather_mod        # noqa: E402,F401


__all__ = [
    "ToolNotFoundError",
    "available_tools",
    "invoke",
]
