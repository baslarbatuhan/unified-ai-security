"""
output_agency_defense/error_policy.py
=======================================
Uniform error response policy.

Purpose:
    - Return SAME error message for "not found" and "unauthorized"
    - Format: "unauthorized_or_not_found"
    - Prevents attacker from learning whether a resource exists
    - Non-existent ID and unauthorized ID produce identical responses

OWASP Reference:
    - IDOR Prevention: don't reveal resource existence to unauthorized users
    - Information Disclosure Prevention: uniform error responses

Security Principle:
    If an attacker queries order "ORD-999" (doesn't exist) and gets
    "not found", then queries "ORD-001" (exists but not theirs) and
    gets "unauthorized", they now know ORD-001 exists. The uniform
    error prevents this enumeration attack.
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Uniform error constant
# ---------------------------------------------------------------------------
UNIFORM_ERROR_MESSAGE = "unauthorized_or_not_found"


# ---------------------------------------------------------------------------
# Timing normalization config
# ---------------------------------------------------------------------------
NORMALIZED_DELAY_MS = 50  # minimum response time in milliseconds


def uniform_error() -> str:
    """
    Return the standard uniform error message.
    Used by object_authz_guard and secure_tool_wrapper.
    """
    return UNIFORM_ERROR_MESSAGE


@dataclass
class ErrorResponse:
    """Standardized error response object."""
    error: str = UNIFORM_ERROR_MESSAGE
    status_code: int = 403
    detail: Optional[str] = None  # only populated in debug mode

    def to_dict(self, debug: bool = False) -> dict:
        result = {
            "error": self.error,
            "status_code": self.status_code,
        }
        if debug and self.detail:
            result["detail"] = self.detail
        return result


def create_error_response(
    actual_reason: str = "",
    debug: bool = False,
) -> ErrorResponse:
    """
    Create a uniform error response.

    The actual_reason is stored internally for logging/audit
    but NEVER exposed to the user unless debug mode is on.

    Args:
        actual_reason: Internal reason (for audit log only)
        debug:         If True, include detail in response (NEVER in production)
    """
    return ErrorResponse(
        error=UNIFORM_ERROR_MESSAGE,
        status_code=403,
        detail=actual_reason if debug else None,
    )


def normalize_timing(start_time: float) -> None:
    """
    Normalize response timing to prevent timing side-channel attacks.

    If "not found" returns in 2ms and "unauthorized" returns in 15ms,
    an attacker can distinguish between the two by measuring response time.
    This function ensures all error responses take at least NORMALIZED_DELAY_MS.

    Args:
        start_time: time.time() when the request started processing
    """
    elapsed_ms = (time.time() - start_time) * 1000
    remaining_ms = NORMALIZED_DELAY_MS - elapsed_ms

    if remaining_ms > 0:
        # Add slight jitter to avoid fingerprinting the exact delay
        jitter = random.uniform(0, 5)
        time.sleep((remaining_ms + jitter) / 1000)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"{'='*50}")
    print(f"  UNIFORM ERROR POLICY DEMO")
    print(f"{'='*50}")

    # Scenario 1: Resource not found
    t0 = time.time()
    resp1 = create_error_response("Resource ORD-999 does not exist")
    normalize_timing(t0)
    elapsed1 = (time.time() - t0) * 1000
    print(f"\n  Scenario: Non-existent resource")
    print(f"    User sees:  {resp1.to_dict()}")
    print(f"    Response time: {elapsed1:.1f}ms")

    # Scenario 2: Unauthorized access
    t0 = time.time()
    resp2 = create_error_response("User bob not authorized for ORD-001 owned by alice")
    normalize_timing(t0)
    elapsed2 = (time.time() - t0) * 1000
    print(f"\n  Scenario: Unauthorized access (IDOR attempt)")
    print(f"    User sees:  {resp2.to_dict()}")
    print(f"    Response time: {elapsed2:.1f}ms")

    # Both responses are identical to the attacker
    print(f"\n  Both responses identical: {resp1.to_dict() == resp2.to_dict()}")
    print(f"  Timing difference: {abs(elapsed1 - elapsed2):.1f}ms (should be minimal)")

    # Debug mode (internal only, never in production)
    print(f"\n  [DEBUG MODE - internal audit only]")
    resp_debug = create_error_response("IDOR: bob tried to access alice's order", debug=True)
    print(f"    Debug view: {resp_debug.to_dict(debug=True)}")
