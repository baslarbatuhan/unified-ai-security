"""
evaluation/behavior_weight_calibration.py
=============================================
Behavior Risk Weight Calibration Test.

Tests 3 different weight configurations for behavior monitoring signals
(burst, diversity, repetition, failure, lateral) against the agency
attack scenario dataset.

Output:
    runs/agency_behavior_weight_analysis.csv

Usage:
    python evaluation/behavior_weight_calibration.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_RUNS_DIR = _PROJECT_ROOT / "runs"

# Weight configurations to test
WEIGHT_CONFIGS = {
    "current": {
        "burst": 0.20, "diversity": 0.20, "repetition": 0.15,
        "failure": 0.35, "lateral": 0.10,
    },
    "failure_heavy": {
        "burst": 0.20, "diversity": 0.15, "repetition": 0.10,
        "failure": 0.45, "lateral": 0.10,
    },
    "balanced": {
        "burst": 0.25, "diversity": 0.20, "repetition": 0.15,
        "failure": 0.25, "lateral": 0.15,
    },
}


def load_scenarios() -> List[Dict]:
    """Load agency attack scenarios."""
    path = _PROJECT_ROOT / "datasets" / "output_agency_attacks" / "agency_attack_scenarios.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["scenarios"]


def simulate_with_weights(scenarios: List[Dict], weights: Dict[str, float]) -> Dict:
    """Simulate behavior risk model with given weights.

    For each scenario, simulates behavior signals and calculates risk
    using the given weight configuration.
    """
    from output_agency_defense.behavior_monitor import BehaviorMonitor
    from output_agency_defense.anti_enum_guard import AntiEnumGuard
    from output_agency_defense.parameter_validation import ParameterValidator
    from fusion_gateway.engine import _register_gateway_demo_schemas

    monitor = BehaviorMonitor()
    # Override weights
    monitor._signal_weights = weights

    enum_guard = AntiEnumGuard()
    param_validator = ParameterValidator()
    _register_gateway_demo_schemas(param_validator)

    results = {"total": 0, "blocked": 0, "allowed": 0, "correct": 0}

    for s in scenarios:
        results["total"] += 1
        expected = s["expected_decision"]
        tool = s["tool"]
        args = s["args"]
        user_id = s["user_id"]
        role = s.get("role", "basic")
        resource_id = str(args.get("resource_id", "") or "")

        # Simulate: record behavior event
        was_authorized = expected == "allow"
        assessment = monitor.record(
            user_id=user_id,
            tool=tool,
            resource_id=resource_id,
            resource_type="order" if "ORD" in resource_id else "ticket" if "TKT" in resource_id else "unknown",
            was_authorized=was_authorized,
        )

        # Calculate risk from behavior assessment
        risk_level = assessment.risk_level
        if risk_level in ("critical", "high"):
            decision = "block"
        elif risk_level == "medium":
            decision = "flag"
        else:
            decision = "allow"

        if expected == "block" and decision in ("block", "flag"):
            results["blocked"] += 1
            results["correct"] += 1
        elif expected == "allow" and decision == "allow":
            results["allowed"] += 1
            results["correct"] += 1
        elif expected == "allow":
            results["allowed"] += 1
        else:
            results["blocked"] += 1 if decision in ("block", "flag") else 0

    results["detection_rate"] = results["blocked"] / max(1, sum(1 for s in scenarios if s["expected_decision"] == "block")) * 100
    results["accuracy"] = results["correct"] / max(1, results["total"]) * 100

    return results


def run_calibration():
    """Run calibration across all weight configs."""
    print(f"\n{'='*65}")
    print(f"  BEHAVIOR WEIGHT CALIBRATION")
    print(f"{'='*65}")

    scenarios = load_scenarios()
    print(f"  Loaded {len(scenarios)} scenarios")

    csv_rows = []
    for config_name, weights in WEIGHT_CONFIGS.items():
        results = simulate_with_weights(scenarios, weights)
        print(f"\n  [{config_name}]")
        print(f"    Weights: {weights}")
        print(f"    Detection rate: {results['detection_rate']:.1f}%")
        print(f"    Accuracy: {results['accuracy']:.1f}%")
        print(f"    Blocked: {results['blocked']}, Allowed: {results['allowed']}")

        csv_rows.append({
            "config": config_name,
            "burst_w": weights["burst"],
            "diversity_w": weights["diversity"],
            "repetition_w": weights["repetition"],
            "failure_w": weights["failure"],
            "lateral_w": weights["lateral"],
            "total": results["total"],
            "blocked": results["blocked"],
            "allowed": results["allowed"],
            "correct": results["correct"],
            "detection_rate": round(results["detection_rate"], 2),
            "accuracy": round(results["accuracy"], 2),
        })

    # Save CSV
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RUNS_DIR / "agency_behavior_weight_analysis.csv"
    fieldnames = list(csv_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n  [Saved] {csv_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    run_calibration()
