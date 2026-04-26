"""utils/rate_limiter.py
==========================
Token-bucket rate limiter backed by `configs/service_limits.yaml`.

One bucket per (tier, identity) pair. Tiers map to the top-level keys under
`rate_limit:` in the YAML (default / external_eval / admin). Identity is
whatever the caller uses to partition traffic — user_id for REST, target_id
for the external_eval runner, "global" for single-tenant bursts.

Design notes
------------
* Pure in-memory, thread-safe. Good enough for single-worker gateways and
  every evaluation/test scenario. Swap in a Redis-backed impl later without
  touching call sites — the `acquire()` contract is intentionally minimal.
* Bucket capacity = burst * burst_multiplier (ceil). Refill rate =
  requests_per_minute / 60 tokens per second.
* `acquire()` is non-blocking — callers decide whether to reject (HTTP 429)
  or sleep. That keeps the limiter free of asyncio / threading assumptions.
* Unknown tier → falls back to the "default" profile and logs a warning
  rather than raising, so a typo in a call site doesn't take the gateway
  down.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Optional, Tuple

from configs.timeout_loader import load_service_limits


@dataclass
class _Bucket:
    capacity: float
    refill_per_sec: float
    tokens: float
    last_refill_ts: float


@dataclass
class AcquireResult:
    """Return value of RateLimiter.acquire()."""
    allowed: bool
    remaining: float
    retry_after_sec: float = 0.0
    reason: str = ""
    tier: str = ""

    def to_headers(self) -> Dict[str, str]:
        """HTTP-friendly headers (RateLimit-* + Retry-After on deny)."""
        hdrs = {
            "X-RateLimit-Remaining": str(int(max(0, math.floor(self.remaining)))),
            "X-RateLimit-Tier": self.tier,
        }
        if not self.allowed:
            hdrs["Retry-After"] = str(max(1, int(math.ceil(self.retry_after_sec))))
        return hdrs


class RateLimiter:
    """In-memory token-bucket limiter keyed by (tier, identity)."""

    def __init__(self, config: Optional[dict] = None):
        self._cfg = (config or load_service_limits()).get("rate_limit", {})
        if "default" not in self._cfg:
            raise KeyError("service_limits.yaml: rate_limit.default is required")
        self._buckets: Dict[Tuple[str, str], _Bucket] = {}
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def acquire(self, identity: str, *, tier: str = "default", cost: float = 1.0) -> AcquireResult:
        """Try to consume `cost` tokens for (tier, identity).

        Returns AcquireResult with allowed=False + retry_after_sec when the
        bucket is dry. Never blocks, never raises on unknown tiers.
        """
        effective_tier = tier if tier in self._cfg else "default"
        key = (effective_tier, identity)

        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = self._new_bucket(effective_tier)
                self._buckets[key] = bucket
            self._refill(bucket)

            if bucket.tokens + 1e-9 >= cost:
                bucket.tokens -= cost
                return AcquireResult(
                    allowed=True,
                    remaining=bucket.tokens,
                    tier=effective_tier,
                )

            # Not enough tokens — compute wait for the deficit to refill.
            deficit = cost - bucket.tokens
            retry = deficit / bucket.refill_per_sec if bucket.refill_per_sec > 0 else 60.0
            reason = (
                f"rate_limit_exceeded tier={effective_tier} "
                f"capacity={bucket.capacity:.1f} refill={bucket.refill_per_sec:.3f}/s"
            )
            return AcquireResult(
                allowed=False,
                remaining=max(0.0, bucket.tokens),
                retry_after_sec=retry,
                reason=reason,
                tier=effective_tier,
            )

    def reset(self) -> None:
        """Wipe every bucket — test hook; production code should not call this."""
        with self._lock:
            self._buckets.clear()

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """Diagnostic view: {tier:identity: {tokens, capacity}}."""
        with self._lock:
            return {
                f"{t}:{ident}": {"tokens": b.tokens, "capacity": b.capacity}
                for (t, ident), b in self._buckets.items()
            }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _new_bucket(self, tier: str) -> _Bucket:
        cfg = self._cfg[tier]
        rpm = float(cfg.get("requests_per_minute", 60))
        burst = float(cfg.get("burst", 10))
        mul = float(cfg.get("burst_multiplier", 1.0))
        capacity = max(1.0, math.ceil(burst * mul))
        refill_per_sec = rpm / 60.0 if rpm > 0 else 1.0
        return _Bucket(
            capacity=capacity,
            refill_per_sec=refill_per_sec,
            tokens=capacity,  # start full so first burst is allowed
            last_refill_ts=time.monotonic(),
        )

    @staticmethod
    def _refill(bucket: _Bucket) -> None:
        now = time.monotonic()
        elapsed = max(0.0, now - bucket.last_refill_ts)
        bucket.tokens = min(bucket.capacity, bucket.tokens + elapsed * bucket.refill_per_sec)
        bucket.last_refill_ts = now


# Process-wide singleton — convenience for call sites that don't want to
# thread a limiter instance through the code. Lazy-initialised so tests can
# monkeypatch load_service_limits before first access.
_DEFAULT: Optional[RateLimiter] = None
_DEFAULT_LOCK = Lock()


def default_limiter() -> RateLimiter:
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = RateLimiter()
        return _DEFAULT


def reset_default_limiter() -> None:
    """Test hook — forces re-read of service_limits.yaml on next call."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        _DEFAULT = None


__all__ = [
    "RateLimiter",
    "AcquireResult",
    "default_limiter",
    "reset_default_limiter",
]
