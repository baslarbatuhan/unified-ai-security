"""api/rate_limiter.py
======================
Spec-path re-export of the canonical implementation in
``utils.rate_limiter``. The audit / architecture spec references the
limiter at this path; the actual code lives in ``utils/`` because it is
shared by middleware, dashboard routes, and the external_eval harness.

Importing from either path yields the same module objects — there is no
duplication, just a thin alias.
"""
from __future__ import annotations

from utils.rate_limiter import *  # noqa: F401,F403
from utils.rate_limiter import (  # noqa: F401  (explicit re-export for IDEs / linters)
    RateLimiter,
    default_limiter,
    reset_default_limiter,
)
