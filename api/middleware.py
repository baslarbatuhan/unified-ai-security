"""api/middleware.py
======================
Starlette/FastAPI middlewares wired into `api/api_main.py`.

RateLimitMiddleware
-------------------
Token-bucket limiter applied per-(tier, client). The tier is chosen by
inspecting the request path so admin/dashboard traffic gets a roomier bucket
than /analyze. Identity defaults to the client IP but can be overridden via
`X-User-Id` or `X-Target-Id` (the external-eval runner sets the latter).

On reject we emit a 429 with Retry-After + RateLimit-* headers so the
dashboard can show a meaningful badge instead of a generic failure.
"""

from __future__ import annotations

from typing import Callable, Optional

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from utils.rate_limiter import RateLimiter, default_limiter


# Paths we *never* rate-limit — health probes and the OpenAPI docs must
# respond even when the gateway is under load.
_BYPASS_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


def _tier_for_path(path: str) -> str:
    if path.startswith("/dashboard") or path.startswith("/admin"):
        return "admin"
    if path.startswith("/external-eval"):
        return "external_eval"
    return "default"


def _identity(request: Request) -> str:
    # Explicit headers win so the external-eval runner can stay on its own
    # bucket regardless of where it calls from.
    for hdr in ("X-User-Id", "X-Target-Id"):
        v = request.headers.get(hdr)
        if v:
            return v.strip()[:128]
    if request.client and request.client.host:
        return request.client.host
    return "anonymous"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose token bucket is empty.

    The limiter instance can be injected for tests; in production we use the
    process-wide singleton so every request path shares the same buckets.
    """

    def __init__(self, app, limiter: Optional[RateLimiter] = None,
                 bypass_paths: Optional[set] = None):
        super().__init__(app)
        self._limiter = limiter
        self._bypass = bypass_paths if bypass_paths is not None else _BYPASS_PATHS

    @property
    def limiter(self) -> RateLimiter:
        # Lazy so module import doesn't force a YAML read.
        return self._limiter or default_limiter()

    async def dispatch(self, request: Request, call_next: Callable):
        path = request.url.path
        if path in self._bypass:
            return await call_next(request)

        tier = _tier_for_path(path)
        identity = _identity(request)
        result = self.limiter.acquire(identity, tier=tier)
        if not result.allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "rate_limit_exceeded",
                    "tier": result.tier,
                    "retry_after_sec": round(result.retry_after_sec, 2),
                    "reason": result.reason,
                },
                headers=result.to_headers(),
            )
        response = await call_next(request)
        # Attach remaining-budget headers so the dashboard can show a meter.
        for k, v in result.to_headers().items():
            response.headers[k] = v
        return response


__all__ = ["RateLimitMiddleware"]
