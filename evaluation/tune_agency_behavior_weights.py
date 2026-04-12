"""
evaluation/tune_agency_behavior_weights.py
=============================================
Agency Behavior Weight Tuning via Grid Search.

Extends behavior_weight_calibration.py with a finer grid search
over 5 signal weights (burst, diversity, repetition, failure, lateral)
to find the optimal configuration.

Constraints:
    - All weights must sum to 1.0
    - Each weight in [0.05, 0.50]

Output:
    runs/agency_behavior_weight_tuning.csv

Usage:
    python evaluation/tune_agency_behavior_weights.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_RUNS_DIR = _PROJECT_ROOT / "runs"


def load_scenarios() -> List[Dict]:
    """Load agency attack scenarios."""
    path = _PROJECT_ROOT / "datasets" / "output_agency_attacks" / "agency_attack_scenarios.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["scenarios"]


def generate_weight_grid() -> List[Dict[str, float]]:
    """Generate weight combinations that sum to 1.0.

    Uses a coarse grid (step=0.05) to keep computation manageable.
    """
    step = 0.05
    signals = ["burst", "diversity", "repetition", "failure", "lateral"]
    values = [round(v, 2) for v in [i * step for i in range(1, 11)]]  # 0.05 to 0.50

    grid = []
    for burst in values:
        for diversity in values:
            for repetition in values:
                for failure in values:
                    lateral = round(1.0 - burst - diversity - repetition - failure, 2)
                    if 0.05 <= lateral <= 0.50:
                        grid.append({
                            "burst": burst,
                            "diversity": diversity,
                            "repetition": repetition,
                            "failure": failure,
                            "lateral": lateral,
                        })
    return grid


def evaluate_weights(scenarios: List[Dict], weights: Dict[str, float]) -> Dict:
    """Evaluate a weight configuration against scenarios."""
    from output_agency_defense.behavior_monitor import BehaviorMonitor

    monitor = BehaviorMonitor()
    monitor._signal_weights = weights

    tp, fp, tn, fn = 0, 0, 0, 0

    for s in scenarios:
        expected = s["expected_decision"]
        resource_id = str(s["args"].get("resource_id", "") or "")

        assessment = monitor.record(
            user_id=s["user_id"],
            tool=s["tool"],
            resource_id=resource_id,
            resource_type="order" if "ORD" in resource_id else "ticket" if "TKT" in resource_id else "unknown",
            was_authorized=(expected == "allow"),
        )

        risk_level = assessment.risk_level
        detected = risk_level in ("critical", "high", "medium")

        is_attack = expected == "block"
        if is_attack and detected:
            tp += 1
        elif is_attack and not detected:
            fn += 1
        elif not is_attack and detected:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def run_tuning():
    """Run grid search over weight configurations."""
    print(f"\n{'='*65}")
    print(f"  AGENCY BEHAVIOR WEIGHT TUNING (Grid Search)")
    print(f"{'='*65}")

    scenarios = load_scenarios()
    print(f"  Loaded {len(scenarios)} scenarios")

    grid = generate_weight_grid()
    print(f"  Generated {len(grid)} weight combinations")

    if len(grid) > 5000:
        # Sample to keep runtime reasonable
        import random
        random.seed(42)
        grid = random.sample(grid, 5000)
        print(f"  Sampled down to {len(grid)} combinations")

    print(f"\n  Running grid search...")
    t0 = time.time()

    best_f1 = 0.0
    best_config = None
    csv_rows = []

    for i, weights in enumerate(grid):
        metrics = evaluate_weights(scenarios, weights)
        row = {**weights, **metrics}
        csv_rows.append(row)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_config = row

        if (i + 1) % 1000 == 0:
            print(f"    Evaluated {i+1}/{len(grid)}... (best F1={best_f1:.4f})")

    elapsed = int((time.time() - t0) * 1000)
    print(f"\n  Grid search complete in {elapsed}ms")

    # Save top 50 results
    csv_rows.sort(key=lambda r: r["f1"], reverse=True)
    top_results = csv_rows[:50]

    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RUNS_DIR / "agency_behavior_weight_tuning.csv"
    fieldnames = list(top_results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(top_results)
    print(f"  [Saved] {csv_path} (top 50 of {len(grid)})")

    if best_config:
        print(f"\n  BEST CONFIG:")
        print(f"    burst={best_config['burst']:.2f} diversity={best_config['diversity']:.2f} "
              f"repetition={best_config['repetition']:.2f} failure={best_config['failure']:.2f} "
              f"lateral={best_config['lateral']:.2f}")
        print(f"    F1={best_config['f1']:.4f} P={best_config['precision']:.4f} "
              f"R={best_config['recall']:.4f} FPR={best_config['fpr']:.4f}")

    print(f"{'='*65}")
    return top_results


if __name__ == "__main__":
    run_tuning()
