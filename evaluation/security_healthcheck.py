"""
evaluation/security_healthcheck.py
=====================================
System security healthcheck script.

Purpose:
    Reports:
    - active guards
    - tool coverage
    - anti-enumeration status
    - error policy status

Usage:
    python evaluation/security_healthcheck.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

_FILE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FILE_DIR.parent if _FILE_DIR.name == "evaluation" else _FILE_DIR
_RUNS_DIR = _PROJECT_ROOT / "runs"

sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from output_agency_defense.resource_registry import create_demo_registry
    from output_agency_defense.object_authz_guard import ObjectAuthzGuard, Session
    from output_agency_defense.anti_enum_guard import AntiEnumGuard
    from output_agency_defense.guard_registry import GuardRegistry, REQUIRED_GUARDS
    from output_agency_defense.coverage_check import ToolCoverageChecker
    from output_agency_defense.error_policy import UNIFORM_ERROR_MESSAGE, uniform_error
    from output_agency_defense.secure_tool_wrapper import SecureToolWrapper
except ImportError:
    sys.path.insert(0, str(_PROJECT_ROOT / "output_agency_defense"))
    from resource_registry import create_demo_registry
    from object_authz_guard import ObjectAuthzGuard, Session
    from anti_enum_guard import AntiEnumGuard
    from guard_registry import GuardRegistry, REQUIRED_GUARDS
    from coverage_check import ToolCoverageChecker
    from error_policy import UNIFORM_ERROR_MESSAGE, uniform_error
    from secure_tool_wrapper import SecureToolWrapper


# ---------------------------------------------------------------------------
# Healthcheck result
# ---------------------------------------------------------------------------
@dataclass
class HealthcheckResult:
    timestamp: str = ""
    overall_status: str = "UNKNOWN"  # HEALTHY, DEGRADED, CRITICAL
    checks: List[Dict] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Healthcheck runner
# ---------------------------------------------------------------------------
def run_healthcheck() -> HealthcheckResult:
    """
    Run all security healthchecks and produce a report.

    Checks:
    1. Guard registry — required guards active
    2. Tool wrapper — enabled and functional
    3. Tool coverage — all tools go through wrapper
    4. Anti-enumeration — detector operational
    5. Error policy — uniform error active
    """
    result = HealthcheckResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    passed = 0
    failed = 0
    total = 5

    # --- Setup system components ---
    registry = create_demo_registry()
    authz = ObjectAuthzGuard(registry)
    enum_guard = AntiEnumGuard()

    guard_reg = GuardRegistry()
    guard_reg.register("object_authz", authz, description="IDOR prevention")
    guard_reg.register("anti_enum", enum_guard, description="Anti-enumeration")

    wrapper = SecureToolWrapper(registry, authz, enabled=True)
    wrapper.register_tool("get_order", lambda **kw: {}, requires_resource=True, resource_type="order")
    wrapper.register_tool("cancel_order", lambda **kw: {}, requires_resource=True, resource_type="order")
    wrapper.register_tool("get_ticket", lambda **kw: {}, requires_resource=True, resource_type="ticket")
    wrapper.register_tool("system_status", lambda **kw: {"status": "ok"}, requires_resource=False)

    # --- Check 1: Guard Registry ---
    try:
        guard_validation = guard_reg.validate()
        active_guards = guard_reg.list_active()
        result.checks.append({
            "name": "active_guards",
            "status": "PASS",
            "detail": f"All required guards active: {active_guards}",
            "required": REQUIRED_GUARDS,
            "active": active_guards,
        })
        passed += 1
    except RuntimeError as e:
        result.checks.append({
            "name": "active_guards",
            "status": "FAIL",
            "detail": str(e),
            "required": REQUIRED_GUARDS,
            "active": guard_reg.list_active(),
        })
        failed += 1

    # --- Check 2: Tool Wrapper ---
    wrapper_ok = wrapper.is_enabled
    if wrapper_ok:
        # Verify it actually blocks when needed
        test_result = wrapper.invoke("get_order", {"resource_id": "ORD-001"},
                                     Session(user="unauthorized_user", role="basic"))
        wrapper_blocks = test_result["status"] == "blocked"
    else:
        wrapper_blocks = False

    if wrapper_ok and wrapper_blocks:
        result.checks.append({
            "name": "tool_wrapper",
            "status": "PASS",
            "detail": "Wrapper enabled and correctly blocking unauthorized access",
        })
        passed += 1
    else:
        result.checks.append({
            "name": "tool_wrapper",
            "status": "FAIL",
            "detail": f"Wrapper enabled={wrapper_ok}, blocks unauthorized={wrapper_blocks}",
        })
        failed += 1

    # --- Check 3: Tool Coverage ---
    checker = ToolCoverageChecker()
    known_tools = ["get_order", "cancel_order", "get_ticket", "system_status"]
    for t in known_tools:
        checker.register_known_tool(t)

    coverage = checker.check_coverage(wrapper.list_tools())
    if coverage.is_fully_covered:
        result.checks.append({
            "name": "tool_coverage",
            "status": "PASS",
            "detail": f"All {coverage.total_known_tools} tools covered by wrapper",
            "coverage_ratio": coverage.coverage_ratio,
        })
        passed += 1
    else:
        result.checks.append({
            "name": "tool_coverage",
            "status": "FAIL",
            "detail": f"Coverage gap: {coverage.uncovered} not wrapped",
            "coverage_ratio": coverage.coverage_ratio,
            "uncovered": coverage.uncovered,
        })
        failed += 1

    # --- Check 4: Anti-Enumeration ---
    enum_guard.reset_all()
    # Simulate sequential probe — should detect
    for i in range(1001, 1005):
        probe = enum_guard.check("test_probe_user", f"ORD-{i}")
    enum_works = probe.is_enumeration
    enum_guard.reset_all()

    if enum_works:
        result.checks.append({
            "name": "anti_enumeration_status",
            "status": "PASS",
            "detail": "Sequential probe detection operational",
        })
        passed += 1
    else:
        result.checks.append({
            "name": "anti_enumeration_status",
            "status": "FAIL",
            "detail": "Sequential probe detection NOT working",
        })
        failed += 1

    # --- Check 5: Error Policy ---
    error_msg = uniform_error()
    error_correct = error_msg == UNIFORM_ERROR_MESSAGE

    if error_correct:
        result.checks.append({
            "name": "error_policy_status",
            "status": "PASS",
            "detail": f"Uniform error active: '{error_msg}'",
        })
        passed += 1
    else:
        result.checks.append({
            "name": "error_policy_status",
            "status": "FAIL",
            "detail": f"Error policy misconfigured: got '{error_msg}'",
        })
        failed += 1

    # --- Overall ---
    if failed == 0:
        result.overall_status = "HEALTHY"
    elif failed <= 2:
        result.overall_status = "DEGRADED"
    else:
        result.overall_status = "CRITICAL"

    result.summary = {
        "total_checks": total,
        "passed": passed,
        "failed": failed,
        "status": result.overall_status,
    }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    result = run_healthcheck()

    print(f"\n{'='*60}")
    print(f"  SECURITY HEALTHCHECK REPORT")
    print(f"  {result.timestamp}")
    print(f"{'='*60}")

    for check in result.checks:
        icon = "PASS" if check["status"] == "PASS" else "FAIL"
        print(f"\n  [{icon}] {check['name']}")
        print(f"    {check['detail']}")

    status_color = {"HEALTHY": "OK", "DEGRADED": "WARNING", "CRITICAL": "ERROR"}
    print(f"\n{'='*60}")
    print(f"  OVERALL: {result.overall_status} "
          f"({result.summary['passed']}/{result.summary['total_checks']} checks passed)")
    print(f"{'='*60}")

    # Save report
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _RUNS_DIR / "security_healthcheck.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\n  [Saved] {report_path}")


if __name__ == "__main__":
    main()
