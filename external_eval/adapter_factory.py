"""external_eval/adapter_factory.py
====================================
Single entry point for constructing a `ChatbotAdapter` from a `TargetConfig`.

The runner and the dashboard both use this so they stay agnostic of the
concrete adapter class.

    adapter = build_adapter(target)
    try:
        response = adapter.send(prompt)
    finally:
        adapter.close()
"""

from __future__ import annotations

from schemas.target_schema import TargetConfig
from external_eval.base_adapter import AdapterConfigError, ChatbotAdapter


def build_adapter(target: TargetConfig) -> ChatbotAdapter:
    if not target.enabled:
        raise AdapterConfigError(f"target {target.id!r} is disabled")

    if target.type == "api":
        from external_eval.api_adapter import APIAdapter
        return APIAdapter(target)
    if target.type == "web":
        from external_eval.web_adapter import WebAdapter
        return WebAdapter(target)
    if target.type == "mock":
        from external_eval.mock_adapter import MockAdapter
        return MockAdapter(target)
    if target.type == "tools_local":
        # Hafta 14: tool execution runs gateway-side. The adapter is a
        # no-op placeholder so the runner's target-selection path stays
        # uniform; actual invocation happens via `tools.invoke()`.
        from external_eval.tool_adapter import ToolAdapter
        return ToolAdapter(target)

    raise AdapterConfigError(f"unknown target type {target.type!r}")


__all__ = ["build_adapter"]
