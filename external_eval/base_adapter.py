"""external_eval/base_adapter.py
=================================
Common contract for every chatbot adapter.

The runner knows nothing about whether a target is a REST API, a web page
driven by Playwright, or an in-process mock.  It calls `adapter.send(...)`
and gets back a `ChatbotResponse`.  That's the whole interface.

All adapters:
  * surface timeouts as `AdapterTimeout`
  * surface transport errors as `AdapterTransportError`
  * never raise on empty / garbage responses — they return a valid
    `ChatbotResponse` with `ok=False` and the raw error in `error_message`.
    The runner is responsible for deciding whether `ok=False` counts as a
    failed attack or a broken test.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from schemas.target_schema import TargetConfig


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class AdapterError(Exception):
    """Base class for adapter failures."""


class AdapterTimeout(AdapterError):
    pass


class AdapterTransportError(AdapterError):
    pass


class AdapterConfigError(AdapterError):
    """The target config is invalid for this adapter type."""


# ---------------------------------------------------------------------------
# Response container
# ---------------------------------------------------------------------------
@dataclass
class ChatbotResponse:
    text: str
    ok: bool = True
    latency_ms: int = 0
    target_id: str = ""
    # Adapter-specific fields (status code, selectors matched, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Abstract adapter
# ---------------------------------------------------------------------------
class ChatbotAdapter(abc.ABC):
    """Base class.  Subclasses implement `_send_impl`; the base handles
    latency measurement and consistent error wrapping."""

    def __init__(self, target: TargetConfig):
        self.target = target

    @property
    def target_id(self) -> str:
        return self.target.id

    def send(self, prompt: str, *, session_context: Optional[Dict[str, Any]] = None) -> ChatbotResponse:
        """Public entry point.  Wraps `_send_impl` with timing and error
        normalization so subclasses can be minimal."""
        t0 = time.time()
        try:
            text, metadata = self._send_impl(prompt, session_context or {})
            return ChatbotResponse(
                text=text,
                ok=True,
                latency_ms=int((time.time() - t0) * 1000),
                target_id=self.target.id,
                metadata=metadata,
            )
        except AdapterTimeout as exc:
            return ChatbotResponse(
                text="",
                ok=False,
                latency_ms=int((time.time() - t0) * 1000),
                target_id=self.target.id,
                metadata={"error_class": "timeout"},
                error_message=str(exc),
            )
        except AdapterError as exc:
            return ChatbotResponse(
                text="",
                ok=False,
                latency_ms=int((time.time() - t0) * 1000),
                target_id=self.target.id,
                metadata={"error_class": type(exc).__name__},
                error_message=str(exc),
            )
        except Exception as exc:  # unexpected
            return ChatbotResponse(
                text="",
                ok=False,
                latency_ms=int((time.time() - t0) * 1000),
                target_id=self.target.id,
                metadata={"error_class": "unexpected", "exception_type": type(exc).__name__},
                error_message=str(exc),
            )

    def close(self) -> None:  # pragma: no cover — default no-op
        """Override to release resources (HTTP sessions, browser contexts)."""

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def _send_impl(
        self, prompt: str, session_context: Dict[str, Any]
    ) -> "tuple[str, Dict[str, Any]]":
        """Return `(response_text, metadata_dict)` or raise `AdapterError`."""


__all__ = [
    "AdapterError",
    "AdapterTimeout",
    "AdapterTransportError",
    "AdapterConfigError",
    "ChatbotResponse",
    "ChatbotAdapter",
]
