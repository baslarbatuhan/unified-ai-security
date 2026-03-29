"""
tests/test_id_enumeration.py
================================
ID enumeration test scenarios.

Test Cases:
    1. User accesses own order_id → allow
    2. order_id + 1 access → block
    3. Random ID → block
    4. 20 consecutive ID attempts → enumeration detection

Metrics:
    - unauthorized_access_block_rate
    - enumeration_detection_rate

Usage:
    python tests/test_id_enumeration.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

_FILE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FILE_DIR.parent if _FILE_DIR.name == "tests" else _FILE_DIR
_RUNS_DIR = _PROJECT_ROOT / "runs"

sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from output_agency_defense.resource_registry import create_demo_registry
    from output_agency_defense.object_authz_guard import ObjectAuthzGuard, Session
    from output_agency_defense.anti_enum_guard import AntiEnumGuard
    from output_agency_defense.sequential_probe_detector import SequentialProbeDetector
    from output_agency_defense.secure_tool_wrapper import SecureToolWrapper
except ImportError:
    sys.path.insert(0, str(_PROJECT_ROOT / "output_agency_defense"))
    from resource_registry import create_demo_registry
    from object_authz_guard import ObjectAuthzGuard, Session
    from anti_enum_guard import AntiEnumGuard
    from sequential_probe_detector import SequentialProbeDetector
    from secure_tool_wrapper import SecureToolWrapper


# ---------------------------------------------------------------------------
# Demo tool handlers
# ---------------------------------------------------------------------------
def get_order_handler(resource_id: str, **kw):
    return {"action": "get_order", "resource_id": resource_id}


def system_status_handler(**kw):
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
def run_id_enumeration_tests() -> Dict:
    """
    Run ID enumeration and IDOR test scenarios.

    Returns dict with metrics and per-test results.
    """
    # Setup
    registry = create_demo_registry()
    authz = ObjectAuthzGuard(registry)
    enum_guard = AntiEnumGuard(window_seconds=60, seq_threshold=3, rate_threshold=10)

    # Demo data reminder:
    # ORD-001: owner=user_alice
    # ORD-002: owner=user_bob
    # ORD-003: owner=user_alice
    # ORD-004: owner=user_charlie

    alice = Session(user="user_alice", role="basic")
    bob = Session(user="user_bob", role="basic")
    attacker = Session(user="attacker_eve", role="basic")

    results = []
    unauthorized_blocked = 0
    unauthorized_total = 0
    enum_detected = 0
    enum_total = 0

    print(f"\n{'='*65}")
    print(f"  ID ENUMERATION TEST SUITE")
    print(f"{'='*65}")

    # ========================================
    # TEST 1: User accesses own order → allow
    # ========================================
    print(f"\n  [TEST 1] User accesses own resource → expect ALLOW")

    own_access_tests = [
        (alice, "order", "ORD-001", "Alice accesses her order"),
        (bob, "order", "ORD-002", "Bob accesses his order"),
        (alice, "order", "ORD-003", "Alice accesses her other order"),
    ]

    for session, rtype, rid, desc in own_access_tests:
        authz_result = authz.authorize(rtype, rid, session)
        decision = "allow" if authz_result.is_allowed else "block"
        correct = authz_result.is_allowed
        print(f"    [{decision.upper():5s}] {desc} | correct={correct}")
        results.append({
            "module": "output_agency",
            "test_case": desc,
            "decision": decision,
            "risk_score": 0.0 if correct else 0.9,
            "latency": 0,
            "user_id": session.user,
            "tool": "get_order",
            "resource_id": rid,
            "block_reason": "" if correct else "unexpected_block",
            "test_category": "own_access",
            "expected": "allow",
            "correct": correct,
        })

    # ========================================
    # TEST 2: order_id + 1 access → block
    # ========================================
    print(f"\n  [TEST 2] Adjacent ID access (IDOR) → expect BLOCK")

    idor_tests = [
        (alice, "order", "ORD-002", "Alice tries Bob's order (ORD-001+1)"),
        (bob, "order", "ORD-001", "Bob tries Alice's order"),
        (bob, "order", "ORD-003", "Bob tries Alice's other order"),
        (alice, "order", "ORD-004", "Alice tries Charlie's order"),
    ]

    for session, rtype, rid, desc in idor_tests:
        authz_result = authz.authorize(rtype, rid, session)
        decision = "allow" if authz_result.is_allowed else "block"
        blocked = not authz_result.is_allowed
        unauthorized_total += 1
        if blocked:
            unauthorized_blocked += 1
        print(f"    [{decision.upper():5s}] {desc} | blocked={blocked}")
        results.append({
            "module": "output_agency",
            "test_case": desc,
            "decision": decision,
            "risk_score": 0.9 if blocked else 0.0,
            "latency": 0,
            "user_id": session.user,
            "tool": "get_order",
            "resource_id": rid,
            "block_reason": "idor" if blocked else "",
            "test_category": "idor",
            "expected": "block",
            "correct": blocked,
        })

    # ========================================
    # TEST 3: Random ID → block
    # ========================================
    print(f"\n  [TEST 3] Random non-existent IDs → expect BLOCK")

    random_ids = ["ORD-999", "ORD-5555", "ORD-0000", "TKT-999"]
    for rid in random_ids:
        rtype = "ticket" if rid.startswith("TKT") else "order"
        authz_result = authz.authorize(rtype, rid, attacker)
        decision = "allow" if authz_result.is_allowed else "block"
        blocked = not authz_result.is_allowed
        unauthorized_total += 1
        if blocked:
            unauthorized_blocked += 1
        desc = f"Attacker tries random ID {rid}"
        print(f"    [{decision.upper():5s}] {desc}")
        results.append({
            "module": "output_agency",
            "test_case": desc,
            "decision": decision,
            "risk_score": 0.6 if blocked else 0.0,
            "latency": 0,
            "user_id": attacker.user,
            "tool": "get_order",
            "resource_id": rid,
            "block_reason": "not_found" if blocked else "",
            "test_category": "random_id",
            "expected": "block",
            "correct": blocked,
        })

    # ========================================
    # TEST 4: 20 consecutive ID attempts → enumeration detection
    # ========================================
    print(f"\n  [TEST 4] 20 consecutive ID attempts → expect ENUMERATION DETECTED")

    enum_guard.reset_all()
    enum_total = 20
    first_detection_at = None

    for i in range(1001, 1021):
        rid = f"ORD-{i}"
        enum_result = enum_guard.check(attacker.user, rid)
        decision = enum_result.decision
        is_enum = enum_result.is_enumeration

        if is_enum:
            enum_detected += 1
            if first_detection_at is None:
                first_detection_at = i - 1000

        status = "ENUM_DETECTED" if is_enum else "ok"
        if i <= 1005 or i >= 1018 or is_enum:
            print(f"    ORD-{i}: [{status:14s}] seq={enum_result.sequential_count} | "
                  f"unique={enum_result.unique_attempts} | risk={enum_result.risk_score:.2f}")

        results.append({
            "module": "output_agency",
            "test_case": f"Sequential probe ORD-{i}",
            "decision": decision,
            "risk_score": enum_result.risk_score,
            "latency": 0,
            "user_id": attacker.user,
            "tool": "get_order",
            "resource_id": rid,
            "block_reason": "enumeration" if is_enum else "",
            "test_category": "sequential_probe",
            "expected": "block" if i >= 1003 else "allow",
            "correct": is_enum if i >= 1003 else not is_enum,
        })

    # ========================================
    # METRICS
    # ========================================
    unauthorized_block_rate = unauthorized_blocked / unauthorized_total if unauthorized_total > 0 else 0
    enumeration_detection_rate = enum_detected / enum_total if enum_total > 0 else 0
    total_correct = sum(1 for r in results if r["correct"])
    total_tests = len(results)
    accuracy = total_correct / total_tests if total_tests > 0 else 0

    print(f"\n{'='*65}")
    print(f"  ID ENUMERATION TEST METRICS")
    print(f"{'='*65}")
    print(f"  Total tests:                    {total_tests}")
    print(f"  Correct decisions:              {total_correct}/{total_tests} ({accuracy:.1%})")
    print(f"  Unauthorized access block rate: {unauthorized_block_rate:.1%} ({unauthorized_blocked}/{unauthorized_total})")
    print(f"  Enumeration detection rate:     {enumeration_detection_rate:.1%} ({enum_detected}/{enum_total})")
    if first_detection_at:
        print(f"  First enumeration detected at:  attempt #{first_detection_at}")
    print(f"{'='*65}")

    # ---- SAVE CSV ----
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RUNS_DIR / "agency_metrics.csv"
    fieldnames = [
        "module", "test_case", "decision", "risk_score", "latency",
        "user_id", "tool", "resource_id", "block_reason",
        "test_category", "expected", "correct",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  [Saved] {csv_path}")

    return {
        "total_tests": total_tests,
        "accuracy": round(accuracy, 4),
        "unauthorized_access_block_rate": round(unauthorized_block_rate, 4),
        "enumeration_detection_rate": round(enumeration_detection_rate, 4),
        "first_detection_attempt": first_detection_at,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    metrics = run_id_enumeration_tests()

    if metrics:
        summary_path = _RUNS_DIR / "agency_test_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"  [Saved] {summary_path}")


if __name__ == "__main__":
    main()
