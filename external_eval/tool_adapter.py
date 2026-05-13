"""external_eval/tool_adapter.py
==================================
Adapter for `tools_local` targets — the gateway runs tools itself
instead of forwarding to a remote chatbot.

The chatbot adapter pattern doesn't map cleanly onto tool execution:
  * Chatbot adapter: `send(prompt) → text` (one shape, free-form text)
  * Tool execution: `invoke(tool_name, args) → typed dict`

We keep the `ChatbotAdapter` interface for compatibility with the
runner (so dashboard target-selection works), but `_send_impl` here is
intentionally a no-op that returns an empty placeholder. The actual
tool invocation is handled by the runner via `tools.invoke(...)` AFTER
the gateway has cleared the call. This keeps a clean separation:

    chatbot adapter  ──────► remote chat surface (prompt → text)
    tool adapter     ──────► local capability (tool_call → result)

The runner detects `target.type == "tools_local"` and switches branches.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from external_eval.base_adapter import AdapterConfigError, ChatbotAdapter
from schemas.target_schema import TargetConfig


class ToolAdapter(ChatbotAdapter):
    """No-op adapter for `tools_local` targets.

    `send()` returns an empty placeholder — the runner skips this step
    when it sees a tools_local target and goes straight to
    `tools.invoke()` post-gateway. The class exists so the factory
    dispatch + dashboard target dropdown work uniformly across types.
    """

    def __init__(self, target: TargetConfig):
        super().__init__(target)
        if target.type != "tools_local":
            raise AdapterConfigError(
                f"ToolAdapter requires type='tools_local', got {target.type!r}"
            )

    def _send_impl(
        self, prompt: str, session_context: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """No remote call — return an empty placeholder. The runner
        ignores this output for tools_local targets and invokes the
        tool directly after the gateway pre-screen."""
        return ("", {"note": "tool execution handled by runner, not adapter"})


__all__ = ["ToolAdapter"]
