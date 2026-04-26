"""utils/fallback_handler.py
==============================
Per-dependency circuit breaker with optional fallback callable.

Use one CircuitBreaker per *downstream* (LLM judge, embedding service,
chatbot adapter, …). When the dependency starts failing consistently the
breaker trips OPEN, short-circuits calls for `open_cooldown_seconds`, then
probes with a small number of requests (HALF_OPEN) before closing again.

Thresholds come from `configs/service_limits.yaml:circuit_breaker:` but can
be overridden per-instance — the LLM judge breaker may want a higher
failure_threshold than, say, a flaky web adapter.

State machine (standard three-state breaker):

    CLOSED   --[N consecutive failures]-->   OPEN
    OPEN     --[cooldown elapsed]-->         HALF_OPEN
    HALF_OPEN --[K consecutive successes]-->  CLOSED
    HALF_OPEN --[any failure]-->             OPEN  (cooldown restarts)

Thread-safe via a single lock. The wrapped call itself runs *outside* the
lock so slow dependencies don't serialise the whole process.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, Optional

from configs.timeout_loader import load_service_limits


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised by `call()` when the breaker is OPEN and no fallback is set."""

    def __init__(self, name: str, opened_at: float, cooldown_s: float):
        remaining = max(0.0, cooldown_s - (time.monotonic() - opened_at))
        super().__init__(
            f"Circuit '{name}' is open; retry in ~{remaining:.1f}s"
        )
        self.name = name
        self.retry_after_sec = remaining


@dataclass
class BreakerStats:
    """Diagnostic snapshot."""
    state: CircuitState
    consecutive_failures: int
    consecutive_successes: int
    opened_at: Optional[float]
    total_calls: int
    total_failures: int
    total_short_circuits: int


class CircuitBreaker:
    """Per-dependency breaker. Instantiate one per downstream name."""

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: Optional[int] = None,
        open_cooldown_seconds: Optional[float] = None,
        half_open_success_threshold: Optional[int] = None,
        config: Optional[dict] = None,
    ):
        cfg = ((config or load_service_limits()).get("circuit_breaker") or {})
        self.name = name
        self.failure_threshold = int(
            failure_threshold if failure_threshold is not None
            else cfg.get("failure_threshold", 3)
        )
        self.open_cooldown_seconds = float(
            open_cooldown_seconds if open_cooldown_seconds is not None
            else cfg.get("open_cooldown_seconds", 30)
        )
        self.half_open_success_threshold = int(
            half_open_success_threshold if half_open_success_threshold is not None
            else cfg.get("half_open_success_threshold", 2)
        )

        self._state: CircuitState = CircuitState.CLOSED
        self._fail_streak = 0
        self._succ_streak = 0
        self._opened_at: Optional[float] = None
        self._total_calls = 0
        self._total_failures = 0
        self._total_short_circuits = 0
        self._lock = Lock()

    # ------------------------------------------------------------------
    # State transitions — all mutations go through these helpers.
    # ------------------------------------------------------------------
    def _transition_to(self, new_state: CircuitState) -> None:
        self._state = new_state
        if new_state == CircuitState.OPEN:
            self._opened_at = time.monotonic()
            self._succ_streak = 0
        elif new_state == CircuitState.CLOSED:
            self._fail_streak = 0
            self._succ_streak = 0
            self._opened_at = None
        elif new_state == CircuitState.HALF_OPEN:
            self._succ_streak = 0

    def _should_probe(self) -> bool:
        if self._state != CircuitState.OPEN or self._opened_at is None:
            return False
        return (time.monotonic() - self._opened_at) >= self.open_cooldown_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def allow(self) -> bool:
        """Pre-flight check — True if a call should be attempted now."""
        with self._lock:
            if self._state == CircuitState.OPEN and self._should_probe():
                self._transition_to(CircuitState.HALF_OPEN)
            return self._state != CircuitState.OPEN

    def record_success(self) -> None:
        with self._lock:
            self._fail_streak = 0
            if self._state == CircuitState.HALF_OPEN:
                self._succ_streak += 1
                if self._succ_streak >= self.half_open_success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.OPEN:
                # Late success arriving while open — safe to ignore; state
                # will advance on the next allow() probe.
                pass

    def record_failure(self) -> None:
        with self._lock:
            self._total_failures += 1
            if self._state == CircuitState.HALF_OPEN:
                # Any failure during probe immediately re-opens the breaker.
                self._transition_to(CircuitState.OPEN)
                self._fail_streak = self.failure_threshold
                return
            self._fail_streak += 1
            if self._state == CircuitState.CLOSED and self._fail_streak >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def call(
        self,
        fn: Callable[..., Any],
        *args: Any,
        fallback: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke `fn(*args, **kwargs)` through the breaker.

        * OPEN → short-circuit: call `fallback(*args, **kwargs)` if given,
          otherwise raise CircuitOpenError.
        * CLOSED or HALF_OPEN → invoke fn; update streaks based on outcome.
        """
        with self._lock:
            self._total_calls += 1
            if self._state == CircuitState.OPEN and self._should_probe():
                self._transition_to(CircuitState.HALF_OPEN)
            if self._state == CircuitState.OPEN:
                self._total_short_circuits += 1
                opened_at = self._opened_at or time.monotonic()
                cooldown = self.open_cooldown_seconds

        if self._state == CircuitState.OPEN:
            if fallback is not None:
                return fallback(*args, **kwargs)
            raise CircuitOpenError(self.name, opened_at, cooldown)

        # Run outside the lock so slow calls don't block peers.
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.record_failure()
            if fallback is not None:
                return fallback(*args, **kwargs)
            raise
        else:
            self.record_success()
            return result

    @property
    def state(self) -> CircuitState:
        with self._lock:
            # Auto-advance OPEN→HALF_OPEN when cooldown has elapsed, so
            # observers don't see stale "open" reports.
            if self._state == CircuitState.OPEN and self._should_probe():
                self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    def stats(self) -> BreakerStats:
        with self._lock:
            return BreakerStats(
                state=self._state,
                consecutive_failures=self._fail_streak,
                consecutive_successes=self._succ_streak,
                opened_at=self._opened_at,
                total_calls=self._total_calls,
                total_failures=self._total_failures,
                total_short_circuits=self._total_short_circuits,
            )

    def reset(self) -> None:
        """Force back to CLOSED — test/admin hook."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)


# ---------------------------------------------------------------------------
# Registry — lazy per-name breakers so modules can grab a shared breaker by
# downstream name without coordinating on construction.
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, CircuitBreaker] = {}
_REG_LOCK = Lock()


def get_breaker(name: str, **overrides: Any) -> CircuitBreaker:
    """Get-or-create a breaker by downstream name."""
    with _REG_LOCK:
        br = _REGISTRY.get(name)
        if br is None:
            br = CircuitBreaker(name, **overrides)
            _REGISTRY[name] = br
        return br


def reset_registry() -> None:
    """Test hook — drops every named breaker."""
    with _REG_LOCK:
        _REGISTRY.clear()


__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "CircuitOpenError",
    "BreakerStats",
    "get_breaker",
    "reset_registry",
]
