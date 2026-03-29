"""
tests/test_agency_attack_scenarios.py
==========================================
Dataset-driven agency attack tests.

Reads agency_attack_scenarios.json and runs each scenario through:
    - object_authz_guard (IDOR check)
    - anti_enum_guard (enumeration check)
    - parameter_validation (param safety)
    - Compares observed_decision with expected_decision

Usage:
    python tests/test_agency_attack_scenarios.py
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
    from output_agency_defense.parameter_validation import ParameterValidator
    from output_agency_defense.secure_tool_wrapper import SecureToolWrapper
except ImportError:
    sys.path.insert(0, str(_PROJECT_ROOT / "output_agency_defense"))
    from resource_registry import create_demo_registry
    from object_authz_guard import ObjectAuthzGuard, Session
    from anti_enum_guard import AntiEnumGuard
    from parameter_validation import ParameterValidator
    from secure_tool_wrapper import SecureToolWrapper


def run_agency_attack_tests() -> Dict:
    """Run dataset-driven agency attack scenarios."""

    dataset_path = _PROJECT_ROOT / "datasets" / "output_agency_attacks" / "agency_attack_scenarios.json"
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        return {"status": "error", "error": "dataset_not_found"}

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    scenarios = dataset["scenarios"]

    print(f"\n{'='*65}")
    print(f"  AGENCY ATTACK SCENARIO TESTS")
    print(f"  Total scenarios: {len(scenarios)}")
    print(f"{'='*65}")

    # Setup
    registry = create_demo_registry()
    authz = ObjectAuthzGuard(registry)
    enum_guard = AntiEnumGuard(window_seconds=60, seq_threshold=3, rate_threshold=10)
    param_validator = ParameterValidator()

    # Register param schemas
    param_validator.register_tool_schema("get_order", {
        "resource_id": {"type": "string", "required": True, "pattern": r"^ORD-\d{1,6}$", "max_length": 50},
    })
    param_validator.register_tool_schema("get_ticket", {
        "resource_id": {"type": "string", "required": True, "pattern": r"^TKT-\d{1,6}$", "max_length": 50},
    })
    param_validator.register_tool_schema("cancel_order", {
        "resource_id": {"type": "string", "required": True, "pattern": r"^ORD-\d{1,6}$", "max_length": 50},
    })

    wrapper = SecureToolWrapper(registry, authz, enabled=True)
    wrapper.register_tool("get_order", lambda **kw: {"data": "ok"}, requires_resource=True, resource_type="order")
    wrapper.register_tool("get_ticket", lambda **kw: {"data": "ok"}, requires_resource=True, resource_type="ticket")
    wrapper.register_tool("cancel_order", lambda **kw: {"data": "ok"}, requires_resource=True, resource_type="order")
    wrapper.register_tool("system_status", lambda **kw: {"status": "ok"}, requires_resource=False)

    results = []
    correct = 0
    incorrect = 0
    category_stats: Dict[str, Dict] = {}
    _enum_reset_done: set = set()

    for scenario in scenarios:
        sid = scenario["id"]
        category = scenario["category"]
        user_id = scenario["user_id"]
        tool = scenario["tool"]
        args = scenario["args"]
        expected = scenario["expected_decision"]
        role = scenario.get("role", "basic")
        resource_id = args.get("resource_id", "")

        if category not in category_stats:
            category_stats[category] = {"total": 0, "correct": 0}
        category_stats[category]["total"] += 1

        # Reset enum_guard state before each enumeration group starts,
        # so prior scenarios don't leak sequential-probe counts.
        if category == "enumeration_sequential" and user_id not in _enum_reset_done:
            enum_guard.reset_user(user_id)
            _enum_reset_done.add(user_id)

        t0 = time.time()
        observed = "allow"
        block_reason = ""
        risk_score = 0.0

        if category == "enumeration_sequential":
            # Isolated enum test: only exercise the enum_guard so authz
            # on non-existent probe IDs doesn't mask the detection result.
            enum_result = enum_guard.check(user_id, resource_id)
            if enum_result.is_blocked:
                observed = "block"
                block_reason = "enumeration"
                risk_score = 1.0
            else:
                observed = "allow"
                risk_score = 0.0
        else:
            # Full pipeline for all other categories
            # Check 1: Parameter validation
            param_result = param_validator.validate(tool, args)
            if not param_result.is_valid:
                observed = "block"
                block_reason = "param_validation"
                risk_score = param_result.risk_contribution
            else:
                # Check 2: Enumeration detection
                if resource_id and isinstance(resource_id, str):
                    enum_result = enum_guard.check(user_id, resource_id)
                    if enum_result.is_blocked:
                        observed = "block"
                        block_reason = "enumeration"
                        risk_score = 1.0
                    else:
                        # Check 3: Tool wrapper (authz + registered tool)
                        session = Session(user=user_id, role=role)
                        wrapper_result = wrapper.invoke(tool, args, session)
                        if wrapper_result.get("status") == "blocked":
                            observed = "block"
                            block_reason = wrapper_result.get("reason", "authz")
                            risk_score = 0.9
                        else:
                            observed = "allow"
                            risk_score = 0.0
                else:
                    # No resource_id — check if tool is registered
                    session = Session(user=user_id, role=role)
                    wrapper_result = wrapper.invoke(tool, args, session)
                    if wrapper_result.get("status") == "blocked":
                        observed = "block"
                        block_reason = wrapper_result.get("reason", "unknown")
                        risk_score = 0.7
                    else:
                        observed = "allow"
                        risk_score = 0.0

        latency_ms = int((time.time() - t0) * 1000)

        # Compare with expected
        is_correct = observed == expected
        if is_correct:
            correct += 1
            category_stats[category]["correct"] += 1

        else:
            incorrect += 1

        status = "PASS" if is_correct else "FAIL"
        print(f"  [{status}] {sid:8s} | {category:25s} | expected={expected:6s} observed={observed:6s} | "
              f"risk={risk_score:.2f} | {scenario['description'][:40]}")

        results.append({
            "module": "output_agency",
            "test_case": sid,
            "decision": observed,
            "risk_score": round(risk_score, 4),
            "latency": latency_ms,
            "category": category,
            "expected_decision": expected,
            "correct": is_correct,
            "block_reason": block_reason,
            "user_id": user_id,
            "tool": tool,
            "description": scenario["description"],
        })

    # --- METRICS ---
    total = correct + incorrect
    accuracy = correct / total if total > 0 else 0

    print(f"\n{'='*65}")
    print(f"  AGENCY ATTACK SCENARIO RESULTS")
    print(f"{'='*65}")
    print(f"  Total: {total} | Correct: {correct} | Incorrect: {incorrect}")
    print(f"  Accuracy: {accuracy:.1%}")

    print(f"\n  PER-CATEGORY:")
    print(f"  {'Category':<25s} {'Total':>5s} {'Correct':>7s} {'Acc%':>6s}")
    print(f"  {'-'*25} {'-'*5} {'-'*7} {'-'*6}")
    for cat, stats in sorted(category_stats.items()):
        cat_acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {cat:<25s} {stats['total']:>5d} {stats['correct']:>7d} {cat_acc:>5.1f}%")

    # Failed scenarios
    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    {f['test_case']}: expected={f['expected_decision']} got={f['decision']} | {f['description']}")

    print(f"{'='*65}")

    # --- SAVE ---
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RUNS_DIR / "agency_attack_metrics.csv"
    fieldnames = ["module", "test_case", "decision", "risk_score", "latency",
                  "category", "expected_decision", "correct", "block_reason",
                  "user_id", "tool", "description"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  [Saved] {csv_path}")

    summary = {
        "total": total, "correct": correct, "incorrect": incorrect,
        "accuracy": round(accuracy, 4),
        "category_breakdown": category_stats,
    }
    summary_path = _RUNS_DIR / "agency_attack_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  [Saved] {summary_path}")

    return summary


if __name__ == "__main__":
    run_agency_attack_tests()
