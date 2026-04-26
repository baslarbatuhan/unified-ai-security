"""external_eval/mock_adapter.py
=================================
In-process chatbot for tests / offline development.

Returns a deterministic "safe response" that echoes the prompt with a
benign preamble, so the evaluation pipeline has something to score without
needing a real model.  Use `targets.yaml` entry `mock_echo`.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from schemas.target_schema import TargetConfig
from external_eval.base_adapter import AdapterConfigError, ChatbotAdapter


class MockAdapter(ChatbotAdapter):
    def __init__(self, target: TargetConfig):
        super().__init__(target)
        if target.type != "mock":
            raise AdapterConfigError(
                f"MockAdapter requires type='mock', got {target.type!r}"
            )

    def _send_impl(
        self, prompt: str, session_context: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        # Deterministic, side-effect-free.  Echoes length so downstream
        # telemetry has a signal that the prompt reached the adapter.
        body = (
            "I'm a mock chatbot; I cannot execute tools or run code. "
            f"You asked ({len(prompt)} chars): {prompt[:200]}"
        )
        return body, {"mock": True, "prompt_chars": len(prompt)}


__all__ = ["MockAdapter"]
