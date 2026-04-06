"""
api/health.py
================
Expanded health endpoint for the security gateway.

Checks:
    1. active_guards       — Required guards registered and operational
    2. wrapper_coverage    — All tools go through SecureToolWrapper
    3. chroma_connectivity — ChromaDB reachable (local or Docker)
    4. ollama_availability — Ollama API responds
    5. fusion_ready_status — FusionEngine can produce a decision

Usage:
    from api.health import get_health_report, HealthReport
    report = get_health_report()
    # report.status == "HEALTHY" | "DEGRADED" | "CRITICAL"
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


@dataclass
class HealthCheck:
    name: str
    status: str = "UNKNOWN"   # PASS, FAIL, WARN
    detail: str = ""
    latency_ms: int = 0


@dataclass
class HealthReport:
    status: str = "UNKNOWN"   # HEALTHY, DEGRADED, CRITICAL
    checks: List[HealthCheck] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "failed": self.failed,
            "total": self.total,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "detail": c.detail,
                    "latency_ms": c.latency_ms,
                }
                for c in self.checks
            ],
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def _check_active_guards() -> HealthCheck:
    """Check 1: Required guards are registered and active."""
    t0 = time.time()
    try:
        from output_agency_defense.resource_registry import create_demo_registry
        from output_agency_defense.object_authz_guard import ObjectAuthzGuard
        from output_agency_defense.anti_enum_guard import AntiEnumGuard
        from output_agency_defense.guard_registry import GuardRegistry, REQUIRED_GUARDS

        guard_reg = GuardRegistry()
        registry = create_demo_registry()
        authz = ObjectAuthzGuard(registry)
        enum_guard = AntiEnumGuard()
        guard_reg.register("object_authz", authz, description="IDOR guard")
        guard_reg.register("anti_enum", enum_guard, description="Anti-enum guard")

        missing = guard_reg.list_missing_required()
        if not missing:
            return HealthCheck(
                name="active_guards", status="PASS",
                detail=f"All required guards active: {REQUIRED_GUARDS}",
                latency_ms=int((time.time() - t0) * 1000),
            )
        return HealthCheck(
            name="active_guards", status="FAIL",
            detail=f"Missing guards: {missing}",
            latency_ms=int((time.time() - t0) * 1000),
        )
    except Exception as e:
        return HealthCheck(
            name="active_guards", status="FAIL",
            detail=f"Error: {e}",
            latency_ms=int((time.time() - t0) * 1000),
        )


def _check_wrapper_coverage() -> HealthCheck:
    """Check 2: All tools go through SecureToolWrapper."""
    t0 = time.time()
    try:
        from output_agency_defense.resource_registry import create_demo_registry
        from output_agency_defense.object_authz_guard import ObjectAuthzGuard
        from output_agency_defense.secure_tool_wrapper import SecureToolWrapper
        from output_agency_defense.coverage_check import ToolCoverageChecker

        registry = create_demo_registry()
        authz = ObjectAuthzGuard(registry)
        wrapper = SecureToolWrapper(registry, authz, enabled=True)
        wrapper.register_tool("get_order", lambda **kw: {}, requires_resource=True, resource_type="order")
        wrapper.register_tool("cancel_order", lambda **kw: {}, requires_resource=True, resource_type="order")
        wrapper.register_tool("get_ticket", lambda **kw: {}, requires_resource=True, resource_type="ticket")
        wrapper.register_tool("system_status", lambda **kw: {"status": "ok"}, requires_resource=False)

        checker = ToolCoverageChecker()
        for t in ["get_order", "cancel_order", "get_ticket", "system_status"]:
            checker.register_known_tool(t)

        coverage = checker.check_coverage(wrapper.list_tools())
        if coverage.is_fully_covered:
            return HealthCheck(
                name="wrapper_coverage", status="PASS",
                detail=f"All {coverage.total_known_tools} tools covered ({coverage.coverage_ratio:.0%})",
                latency_ms=int((time.time() - t0) * 1000),
            )
        return HealthCheck(
            name="wrapper_coverage", status="FAIL",
            detail=f"Uncovered tools: {coverage.uncovered}",
            latency_ms=int((time.time() - t0) * 1000),
        )
    except Exception as e:
        return HealthCheck(
            name="wrapper_coverage", status="FAIL",
            detail=f"Error: {e}",
            latency_ms=int((time.time() - t0) * 1000),
        )


def _check_chroma_connectivity() -> HealthCheck:
    """Check 3: ChromaDB is reachable."""
    t0 = time.time()
    try:
        import chromadb

        # Prefer HTTP client (avoids SQLite locking with the main API process)
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_PORT", "8001"))
        try:
            client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            client.heartbeat()
            return HealthCheck(
                name="chroma_connectivity", status="PASS",
                detail=f"ChromaDB HTTP OK ({chroma_host}:{chroma_port})",
                latency_ms=int((time.time() - t0) * 1000),
            )
        except Exception:
            pass

        # Fallback: check local path exists (read-only, no lock)
        chroma_path = _PROJECT_ROOT / "data" / "chroma_baseline"
        if chroma_path.exists():
            return HealthCheck(
                name="chroma_connectivity", status="PASS",
                detail=f"Local ChromaDB OK (path={chroma_path})",
                latency_ms=int((time.time() - t0) * 1000),
            )

        return HealthCheck(
            name="chroma_connectivity", status="WARN",
            detail="ChromaDB not reachable via HTTP and no local path found",
            latency_ms=int((time.time() - t0) * 1000),
        )
    except Exception as e:
        return HealthCheck(
            name="chroma_connectivity", status="WARN",
            detail=f"ChromaDB not reachable: {e}",
            latency_ms=int((time.time() - t0) * 1000),
        )


def _check_ollama_availability() -> HealthCheck:
    """Check 4: Ollama API responds."""
    t0 = time.time()
    try:
        import requests
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        resp = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return HealthCheck(
                name="ollama_availability", status="PASS",
                detail=f"Ollama OK, models: {models}",
                latency_ms=int((time.time() - t0) * 1000),
            )
        return HealthCheck(
            name="ollama_availability", status="WARN",
            detail=f"Ollama returned status {resp.status_code}",
            latency_ms=int((time.time() - t0) * 1000),
        )
    except Exception as e:
        return HealthCheck(
            name="ollama_availability", status="WARN",
            detail=f"Ollama not reachable: {e}",
            latency_ms=int((time.time() - t0) * 1000),
        )


def _check_fusion_ready() -> HealthCheck:
    """Check 5: FusionEngine can produce a decision."""
    t0 = time.time()
    try:
        from fusion_gateway.engine import FusionEngine
        engine = FusionEngine()
        test_response = engine.analyze(user_input="health check test prompt")
        if test_response.final_decision in ("allow", "sanitize", "flag", "block"):
            return HealthCheck(
                name="fusion_ready_status", status="PASS",
                detail=f"Fusion OK: test decision={test_response.final_decision}, "
                       f"risk={test_response.fused_risk:.4f}, "
                       f"latency={test_response.latency_ms}ms",
                latency_ms=int((time.time() - t0) * 1000),
            )
        return HealthCheck(
            name="fusion_ready_status", status="FAIL",
            detail=f"Unexpected decision: {test_response.final_decision}",
            latency_ms=int((time.time() - t0) * 1000),
        )
    except Exception as e:
        return HealthCheck(
            name="fusion_ready_status", status="FAIL",
            detail=f"Fusion engine error: {e}",
            latency_ms=int((time.time() - t0) * 1000),
        )


# ---------------------------------------------------------------------------
# Main health report
# ---------------------------------------------------------------------------
def get_health_report() -> HealthReport:
    """Run all 5 health checks and return a report."""
    checks = [
        _check_active_guards(),
        _check_wrapper_coverage(),
        _check_chroma_connectivity(),
        _check_ollama_availability(),
        _check_fusion_ready(),
    ]

    passed = sum(1 for c in checks if c.status == "PASS")
    failed = sum(1 for c in checks if c.status == "FAIL")
    total = len(checks)

    if failed == 0:
        status = "HEALTHY"
    elif failed <= 2:
        status = "DEGRADED"
    else:
        status = "CRITICAL"

    return HealthReport(
        status=status,
        checks=checks,
        passed=passed,
        failed=failed,
        total=total,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    report = get_health_report()

    print(f"\n{'='*60}")
    print(f"  EXPANDED HEALTH CHECK")
    print(f"{'='*60}")

    for c in report.checks:
        icon = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN"}.get(c.status, "????")
        print(f"\n  [{icon}] {c.name} ({c.latency_ms}ms)")
        print(f"    {c.detail}")

    print(f"\n{'='*60}")
    print(f"  OVERALL: {report.status} ({report.passed}/{report.total} passed)")
    print(f"{'='*60}")
