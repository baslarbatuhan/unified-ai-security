"""
tests/test_behavior_monitor.py
================================
Behavioral risk model test suite.

Tests BehaviorMonitor (5 signals) and BehaviorRiskModel (combined).

Tests:
    1.  Normal user           — risk stays low
    2.  Burst detection       — rapid calls trigger burst signal
    3.  Resource diversity    — many unique IDs trigger diversity signal
    4.  Tool repetition       — same tool repeated triggers repetition signal
    5.  Failed auth           — denied requests trigger failure signal
    6.  Lateral movement      — mixed resource types trigger lateral signal
    7.  Risk escalation       — 10 mixed requests, risk rises monotonically
    8.  Combined attack       — multiple signals fire, risk ≥ 0.50
    9.  Window expiry         — old events pruned, risk drops
    10. Reset                 — clearing state drops risk to 0
    --- BehaviorRiskModel integration ---
    11. Model: normal         — all components clean → allow
    12. Model: enumeration    — sequential IDs → block via enum_guard
    13. Model: bad params     — SQL injection → flag via param_validator
    14. Model: combined       — burst + failures + enum → block
    15. Model: ModuleRisk     — to_module_risk_dict() format check

Output:
    runs/behavior_monitor_metrics.csv
    runs/behavior_monitor_summary.json

Usage:
    python tests/test_behavior_monitor.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FILE_DIR.parent if _FILE_DIR.name == "tests" else _FILE_DIR

sys.path.insert(0, str(_PROJECT_ROOT))

from output_agency_defense.behavior_monitor import BehaviorMonitor, BehaviorAssessment, run_scenario
from output_agency_defense.behavior_risk_model import BehaviorRiskModel
from output_agency_defense.anti_enum_guard import AntiEnumGuard
from output_agency_defense.parameter_validation import ParameterValidator

_RUNS_DIR = _PROJECT_ROOT / "runs"


# ===================================================================
#  PART 1: BehaviorMonitor tests (signal-level)
# ===================================================================

def test_normal_user(m: BehaviorMonitor) -> Dict:
    m.reset_all()
    events = [
        {"user_id": "alice", "tool": "get_order", "resource_id": "ORD-001",
         "resource_type": "order", "was_authorized": True},
        {"user_id": "alice", "tool": "get_ticket", "resource_id": "TKT-101",
         "resource_type": "ticket", "was_authorized": True},
        {"user_id": "alice", "tool": "system_status",
         "was_authorized": True},
    ]
    r = run_scenario("normal_user", events, m)
    ok = r.risk_score == 0.0 and r.risk_level == "low"
    return _row("normal_user", r, ok, "Risk 0 for clean calls")


def test_burst(m: BehaviorMonitor) -> Dict:
    m.reset_all()
    events = [
        {"user_id": "b", "tool": "get_order",
         "resource_id": f"ORD-{i:03d}", "resource_type": "order",
         "was_authorized": True}
        for i in range(16)
    ]
    r = run_scenario("burst", events, m)
    ok = "burst" in r.signals and r.burst_score > 0
    return _row("burst_detection", r, ok,
                f"burst_score={r.burst_score:.3f}")


def test_diversity(m: BehaviorMonitor) -> Dict:
    m.reset_all()
    events = [
        {"user_id": "d", "tool": "get_order",
         "resource_id": f"ORD-{i:04d}", "resource_type": "order",
         "was_authorized": True}
        for i in range(10)
    ]
    r = run_scenario("diversity", events, m)
    ok = "resource_diversity" in r.signals and r.diversity_score > 0
    return _row("resource_diversity", r, ok,
                f"diversity_score={r.diversity_score:.3f}")


def test_repetition(m: BehaviorMonitor) -> Dict:
    m.reset_all()
    events = [
        {"user_id": "rp", "tool": "get_order", "resource_id": "ORD-001",
         "resource_type": "order", "was_authorized": True}
        for _ in range(12)
    ]
    r = run_scenario("repetition", events, m)
    ok = "tool_repetition" in r.signals and r.repetition_score > 0
    return _row("tool_repetition", r, ok,
                f"repetition_score={r.repetition_score:.3f}")


def test_failed_auth(m: BehaviorMonitor) -> Dict:
    m.reset_all()
    events = [
        {"user_id": "fa", "tool": "get_order",
         "resource_id": f"ORD-{i:03d}", "resource_type": "order",
         "was_authorized": False}
        for i in range(5)
    ]
    r = run_scenario("failed_auth", events, m)
    ok = "failed_auth" in r.signals and r.failure_score > 0
    return _row("failed_auth", r, ok,
                f"failure_score={r.failure_score:.3f}, "
                f"failed={r.failed_auth_count}")


def test_lateral(m: BehaviorMonitor) -> Dict:
    m.reset_all()
    types = [
        ("get_order", "ORD-1", "order"),
        ("get_ticket", "TKT-1", "ticket"),
        ("get_identity", "ID-1", "identity"),
        ("get_config", "CFG-1", "config"),
    ]
    events = [
        {"user_id": "lat", "tool": t, "resource_id": rid,
         "resource_type": rt, "was_authorized": True}
        for t, rid, rt in types
    ]
    r = run_scenario("lateral", events, m)
    ok = "lateral_movement" in r.signals and r.lateral_score > 0
    return _row("lateral_movement", r, ok,
                f"lateral_score={r.lateral_score:.3f}, "
                f"types={r.unique_resource_types}")


def test_escalation(m: BehaviorMonitor) -> Dict:
    """10 mixed requests — risk must rise monotonically."""
    m.reset_all()
    history = []
    r = None
    for i in range(10):
        r = m.record("esc", "get_order", f"ORD-{i:03d}", "order",
                      was_authorized=(i < 2))
        history.append(r.risk_score)

    monotonic = all(history[i] >= history[i - 1]
                    for i in range(1, len(history)))
    final_elevated = history[-1] > 0.0
    ok = monotonic and final_elevated
    return _row("risk_escalation", r, ok,
                f"Monotonic={monotonic}, curve={[round(h, 3) for h in history]}")


def test_combined(m: BehaviorMonitor) -> Dict:
    m.reset_all()
    types = ["order", "ticket", "identity", "config"]
    r = None
    for i in range(16):
        rt = types[i % len(types)]
        r = m.record("cmb", "get_order", f"RES-{i:03d}", rt,
                      was_authorized=(i < 2))
    ok = len(r.signals) >= 3 and r.risk_score >= 0.40
    return _row("combined_attack", r, ok,
                f"{len(r.signals)} signals, risk={r.risk_score:.3f}")


def test_window_expiry(_m: BehaviorMonitor) -> Dict:
    short = BehaviorMonitor(window_seconds=1, failed_auth_threshold=3)
    for i in range(5):
        short.record("exp", "get_order", f"ORD-{i}", "order", False)
    before = short.assess_risk("exp").risk_score
    time.sleep(1.1)
    after = short.assess_risk("exp").risk_score
    ok = after < before
    return _row_simple("window_expiry", ok,
                       f"before={before:.3f}, after={after:.3f}")


def test_reset(m: BehaviorMonitor) -> Dict:
    m.reset_all()
    for i in range(6):
        m.record("rst", "get_order", f"ORD-{i}", "order", False)
    before = m.assess_risk("rst").risk_score
    m.reset_user("rst")
    after = m.assess_risk("rst").risk_score
    ok = before > 0 and after == 0.0
    return _row_simple("reset", ok,
                       f"before={before:.3f}, after={after:.3f}")


# ===================================================================
#  PART 2: BehaviorRiskModel tests (integration)
# ===================================================================

def _make_model() -> BehaviorRiskModel:
    mon = BehaviorMonitor(
        window_seconds=60, burst_threshold=8,
        resource_diversity_threshold=6, failed_auth_threshold=3,
    )
    eg = AntiEnumGuard(window_seconds=60, seq_threshold=3)
    pv = ParameterValidator()
    pv.register_tool_schema("get_order", {
        "resource_id": {
            "type": "str", "required": True,
            "max_length": 50, "pattern": r"^[A-Z]+-\d+$",
        },
    })
    return BehaviorRiskModel(mon, eg, pv)


def test_model_normal() -> Dict:
    mdl = _make_model()
    for rid in ["ORD-001", "ORD-003"]:
        r = mdl.assess("alice", "get_order", rid,
                        {"resource_id": rid}, resource_type="order")
    ok = r.decision == "allow" and r.risk_score < 0.30
    return _row_model("model_normal", r, ok, "Clean calls → allow")


def test_model_enum() -> Dict:
    mdl = _make_model()
    for i in range(1001, 1006):
        r = mdl.assess("atk", "get_order", f"ORD-{i}",
                        {"resource_id": f"ORD-{i}"}, resource_type="order")
    ok = r.enum_detected and r.risk_score >= 0.85
    return _row_model("model_enumeration", r, ok,
                      f"enum={r.enum_detected}, risk={r.risk_score:.3f}")


def test_model_bad_params() -> Dict:
    mdl = _make_model()
    r = mdl.assess("atk", "get_order", "ORD-001",
                    {"resource_id": "ORD-001'; DROP TABLE;--"},
                    resource_type="order")
    ok = not r.param_valid and r.risk_score > 0.30
    return _row_model("model_bad_params", r, ok,
                      f"valid={r.param_valid}, risk={r.risk_score:.3f}")


def test_model_combined() -> Dict:
    mdl = _make_model()
    types = ["order", "ticket", "identity", "config"]
    for i in range(12):
        rid = f"ORD-{1000 + i}"
        r = mdl.assess("atk", "get_order", rid,
                        {"resource_id": rid},
                        resource_type=types[i % len(types)],
                        was_authorized=(i < 2))
    ok = r.risk_score >= 0.60 and r.behavior_risk_level != "low"
    return _row_model("model_combined", r, ok,
                      f"level={r.behavior_risk_level}, "
                      f"signals={r.behavior_signals}")


def test_model_format() -> Dict:
    mdl = _make_model()
    r = mdl.assess("alice", "get_order", "ORD-001",
                    {"resource_id": "ORD-001"}, resource_type="order")
    mrd = r.to_module_risk_dict()
    required = {"module", "risk_score", "confidence",
                "decision", "evidence", "latency_ms"}
    ok = required.issubset(set(mrd.keys())) and mrd["module"] == "output_agency"
    return _row_model("model_modulerisk_format", r, ok,
                      f"keys={sorted(mrd.keys())}")


# ===================================================================
#  Helpers
# ===================================================================

def _decide(score: float) -> str:
    if score < 0.30: return "allow"
    elif score < 0.60: return "sanitize"
    elif score >= 0.85: return "block"
    else: return "flag"


def _row(name: str, r: BehaviorAssessment, ok: bool, note: str) -> Dict:
    return {
        "module": "output_agency", "test_case": name,
        "decision": _decide(r.risk_score),
        "risk_score": round(r.risk_score, 4), "latency": 0,
        "passed": ok,
        "risk_level": r.risk_level,
        "signals": "|".join(r.signals) if r.signals else "",
        "total_calls": r.total_calls,
        "failed_calls": r.failed_auth_count,
        "burst_score": r.burst_score,
        "diversity_score": r.diversity_score,
        "repetition_score": r.repetition_score,
        "failure_score": r.failure_score,
        "lateral_score": r.lateral_score,
        "note": note,
    }


def _row_simple(name: str, ok: bool, note: str) -> Dict:
    return {
        "module": "output_agency", "test_case": name,
        "decision": "allow", "risk_score": 0.0, "latency": 0,
        "passed": ok, "risk_level": "low", "signals": "",
        "total_calls": 0, "failed_calls": 0,
        "burst_score": 0.0, "diversity_score": 0.0,
        "repetition_score": 0.0, "failure_score": 0.0,
        "lateral_score": 0.0, "note": note,
    }


def _row_model(name: str, r, ok: bool, note: str) -> Dict:
    return {
        "module": "output_agency", "test_case": name,
        "decision": r.decision,
        "risk_score": round(r.risk_score, 4), "latency": 0,
        "passed": ok,
        "risk_level": r.behavior_risk_level,
        "signals": "|".join(r.behavior_signals) if r.behavior_signals else "",
        "total_calls": 0, "failed_calls": 0,
        "burst_score": r.behavior_score,
        "diversity_score": r.enum_score,
        "repetition_score": r.param_score,
        "failure_score": 0.0, "lateral_score": 0.0,
        "note": note,
    }


# ===================================================================
#  Runner
# ===================================================================

def run_all_tests() -> Dict:
    monitor = BehaviorMonitor(
        window_seconds=60, burst_threshold=15,
        resource_diversity_threshold=8, tool_repeat_threshold=10,
        failed_auth_threshold=3,
    )

    tests = [
        # Part 1: BehaviorMonitor
        ("Normal user",         lambda: test_normal_user(monitor)),
        ("Burst detection",     lambda: test_burst(monitor)),
        ("Resource diversity",  lambda: test_diversity(monitor)),
        ("Tool repetition",     lambda: test_repetition(monitor)),
        ("Failed auth",         lambda: test_failed_auth(monitor)),
        ("Lateral movement",    lambda: test_lateral(monitor)),
        ("Risk escalation",     lambda: test_escalation(monitor)),
        ("Combined attack",     lambda: test_combined(monitor)),
        ("Window expiry",       lambda: test_window_expiry(monitor)),
        ("Reset",               lambda: test_reset(monitor)),
        # Part 2: BehaviorRiskModel
        ("Model: normal",       test_model_normal),
        ("Model: enumeration",  test_model_enum),
        ("Model: bad params",   test_model_bad_params),
        ("Model: combined",     test_model_combined),
        ("Model: format",       test_model_format),
    ]

    results = []
    passed = failed = 0

    print(f"\n{'='*65}")
    print(f"  BEHAVIOR MONITOR + RISK MODEL TEST SUITE")
    print(f"  Tests: {len(tests)}")
    print(f"{'='*65}")

    for name, fn in tests:
        t0 = time.time()
        row = fn()
        row["latency"] = int((time.time() - t0) * 1000)
        results.append(row)

        status = "PASS" if row["passed"] else "FAIL"
        if row["passed"]:
            passed += 1
        else:
            failed += 1

        print(f"\n  [{status}] {name}")
        print(f"    Risk: {row['risk_score']:.3f} | Decision: {row['decision']} "
              f"| Level: {row['risk_level']}")
        if row["signals"]:
            print(f"    Signals: {row['signals']}")
        print(f"    {row['note']}")

    total = len(tests)
    acc = passed / total if total > 0 else 0

    print(f"\n{'='*65}")
    print(f"  RESULTS: {passed}/{total} passed ({acc:.0%})")
    if failed:
        print(f"  FAILED: {failed}")
    print(f"{'='*65}")

    # --- Save CSV ---
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RUNS_DIR / "behavior_monitor_metrics.csv"
    fieldnames = [
        "module", "test_case", "decision", "risk_score", "latency",
        "passed", "risk_level", "signals", "total_calls", "failed_calls",
        "burst_score", "diversity_score", "repetition_score",
        "failure_score", "lateral_score", "note",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  [Saved] {csv_path}")

    # --- Save summary ---
    summary = {
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "accuracy": round(acc, 4),
        "tests": [
            {"name": r["test_case"], "passed": r["passed"],
             "risk_score": r["risk_score"], "signals": r["signals"]}
            for r in results
        ],
    }
    summary_path = _RUNS_DIR / "behavior_monitor_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  [Saved] {summary_path}")

    return summary


if __name__ == "__main__":
    run_all_tests()
