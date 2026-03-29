"""
api/security_selfcheck.py
============================
FastAPI startup security self-check.

Purpose:
    When FastAPI starts, verify:
    1. Required guards are active (object_authz, anti_enum)
    2. Tool wrapper is enabled
    3. Uniform error policy is active

    If any check fails → raise "security_guards_missing" and prevent startup.

Usage:
    # In FastAPI main.py:
    from api.security_selfcheck import run_startup_check
    
    app = FastAPI()
    
    @app.on_event("startup")
    async def startup():
        run_startup_check(guard_registry, tool_wrapper)

    # Standalone test:
    python api/security_selfcheck.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_FILE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FILE_DIR.parent if _FILE_DIR.name == "api" else _FILE_DIR

sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from output_agency_defense.guard_registry import GuardRegistry, REQUIRED_GUARDS
    from output_agency_defense.error_policy import UNIFORM_ERROR_MESSAGE, uniform_error
except ImportError:
    sys.path.insert(0, str(_PROJECT_ROOT / "output_agency_defense"))
    from guard_registry import GuardRegistry, REQUIRED_GUARDS
    from error_policy import UNIFORM_ERROR_MESSAGE, uniform_error


# ---------------------------------------------------------------------------
# Security error
# ---------------------------------------------------------------------------
SECURITY_ERROR = "security_guards_missing"


@dataclass
class SelfCheckResult:
    passed: bool = False
    checks: List[Dict] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Startup check
# ---------------------------------------------------------------------------
def run_startup_check(
    guard_registry: Optional[GuardRegistry] = None,
    tool_wrapper: Optional[Any] = None,
) -> SelfCheckResult:
    """
    Run security self-check at FastAPI startup.

    Checks:
    1. Required guards active (object_authz, anti_enum)
    2. Tool wrapper enabled
    3. Uniform error policy active

    Raises RuntimeError with "security_guards_missing" if any check fails.
    """
    result = SelfCheckResult()
    all_passed = True

    # --- Check 1: Required guards ---
    if guard_registry is not None:
        missing = guard_registry.list_missing_required()
        if not missing:
            result.checks.append({
                "check": "required_guards",
                "status": "PASS",
                "detail": f"All required guards active: {REQUIRED_GUARDS}",
            })
        else:
            all_passed = False
            result.checks.append({
                "check": "required_guards",
                "status": "FAIL",
                "detail": f"Missing required guards: {missing}",
            })
    else:
        all_passed = False
        result.checks.append({
            "check": "required_guards",
            "status": "FAIL",
            "detail": "Guard registry not provided",
        })

    # --- Check 2: Tool wrapper ---
    if tool_wrapper is not None:
        wrapper_enabled = getattr(tool_wrapper, "is_enabled", False)
        if wrapper_enabled:
            result.checks.append({
                "check": "tool_wrapper",
                "status": "PASS",
                "detail": "Tool wrapper is enabled",
            })
        else:
            all_passed = False
            result.checks.append({
                "check": "tool_wrapper",
                "status": "FAIL",
                "detail": "Tool wrapper is DISABLED — all tool calls will be blocked",
            })
    else:
        all_passed = False
        result.checks.append({
            "check": "tool_wrapper",
            "status": "FAIL",
            "detail": "Tool wrapper not provided",
        })

    # --- Check 3: Uniform error policy ---
    error_msg = uniform_error()
    if error_msg == UNIFORM_ERROR_MESSAGE:
        result.checks.append({
            "check": "uniform_error",
            "status": "PASS",
            "detail": f"Uniform error active: '{error_msg}'",
        })
    else:
        all_passed = False
        result.checks.append({
            "check": "uniform_error",
            "status": "FAIL",
            "detail": f"Error policy misconfigured: expected '{UNIFORM_ERROR_MESSAGE}', got '{error_msg}'",
        })

    # --- Result ---
    result.passed = all_passed

    if not all_passed:
        failed = [c for c in result.checks if c["status"] == "FAIL"]
        detail = "; ".join(c["detail"] for c in failed)
        result.error = SECURITY_ERROR
        raise RuntimeError(f"{SECURITY_ERROR}: {detail}")

    return result


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
def main():
    try:
        from output_agency_defense.resource_registry import create_demo_registry
        from output_agency_defense.object_authz_guard import ObjectAuthzGuard
        from output_agency_defense.anti_enum_guard import AntiEnumGuard
        from output_agency_defense.secure_tool_wrapper import SecureToolWrapper
    except ImportError:
        from resource_registry import create_demo_registry
        from object_authz_guard import ObjectAuthzGuard
        from anti_enum_guard import AntiEnumGuard
        from secure_tool_wrapper import SecureToolWrapper

    print(f"{'='*60}")
    print(f"  FASTAPI STARTUP SECURITY SELF-CHECK")
    print(f"{'='*60}")

    # --- Test 1: No components provided ---
    print(f"\n  [Test 1] No components → should FAIL")
    try:
        run_startup_check(None, None)
    except RuntimeError as e:
        print(f"    ERROR: {e}")

    # --- Test 2: Guard registry without required guards ---
    print(f"\n  [Test 2] Empty guard registry → should FAIL")
    empty_reg = GuardRegistry()
    try:
        run_startup_check(empty_reg, None)
    except RuntimeError as e:
        print(f"    ERROR: {e}")

    # --- Test 3: All components configured correctly ---
    print(f"\n  [Test 3] Full setup → should PASS")
    registry = create_demo_registry()
    authz = ObjectAuthzGuard(registry)
    enum_guard = AntiEnumGuard()

    guard_reg = GuardRegistry()
    guard_reg.register("object_authz", authz, description="IDOR guard")
    guard_reg.register("anti_enum", enum_guard, description="Anti-enum guard")

    wrapper = SecureToolWrapper(registry, authz, enabled=True)

    try:
        result = run_startup_check(guard_reg, wrapper)
        print(f"    PASSED: All {len(result.checks)} checks OK")
        for c in result.checks:
            print(f"      [{c['status']}] {c['check']}: {c['detail']}")
    except RuntimeError as e:
        print(f"    ERROR: {e}")

    # --- Test 4: Wrapper disabled ---
    print(f"\n  [Test 4] Wrapper disabled → should FAIL")
    wrapper.disable()
    try:
        run_startup_check(guard_reg, wrapper)
    except RuntimeError as e:
        print(f"    ERROR: {e}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
