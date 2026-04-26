"""fusion_gateway/fallback_handler.py
======================================
Spec-path re-export of the canonical implementation in
``utils.fallback_handler``. The architecture spec lists the circuit
breaker under ``fusion_gateway/``; the actual code lives in ``utils/``
because the breaker is shared by every downstream-touching module
(LLM judge, embedding service, chatbot adapters), not just the fusion
engine.

Importing from either path yields the same module objects.
"""
from __future__ import annotations

from utils.fallback_handler import *  # noqa: F401,F403
from utils.fallback_handler import (  # noqa: F401
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    get_breaker,
    reset_registry,
)
