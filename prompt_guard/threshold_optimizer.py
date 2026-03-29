"""
prompt_guard/threshold_optimizer.py
=====================================
Threshold optimization for semantic similarity-based prompt injection detection.

Purpose:
    - Test multiple similarity threshold values
    - Compute precision, recall, F1, FPR, accuracy for each threshold
    - Find optimal threshold balancing security and usability
    - Output results to runs/week2_prompt_metrics.csv

Methodology:
    For each threshold in [0.30, 0.35, ..., 0.90]:
        1. Run semantic evaluator on full dataset (leave-one-out for attacks)
        2. Compute confusion matrix
        3. Record all metrics

Output:
    runs/week2_prompt_metrics.csv with columns:
        threshold, tp, fp, tn, fn, precision, recall, f1, fpr, accuracy

Dependencies:
    Requires semantic_evaluator_v1.py

Usage:
    python prompt_guard/threshold_optimizer.py
    python prompt_guard/threshold_optimizer.py --min-thresh 0.40 --max-thresh 0.85 --step 0.01
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import List, Dict

try:
    from prompt_guard.semantic_evaluator_v1 import SemanticEvaluator
except ImportError:
    from semantic_evaluator_v1 import SemanticEvaluator


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FILE_DIR.parent if _FILE_DIR.name == "prompt_guard" else _FILE_DIR
_DATASET_PATH = _PROJECT_ROOT / "datasets" / "injection_prompts" / "injection_dataset_v1.csv"
_OUTPUT_DIR = _PROJECT_ROOT / "runs"


def run_threshold_optimization(
    evaluator: SemanticEvaluator,
    min_thresh: float = 0.30,
    max_thresh: float = 0.90,
    step: float = 0.05,
) -> List[Dict]:
    """
    Run evaluation at multiple thresholds and collect metrics.

    Returns list of metric dicts, one per threshold.
    """
    thresholds = []
    t = min_thresh
    while t <= max_thresh + 0.001:
        thresholds.append(round(t, 3))
        t += step

    print(f"[Optimizer] Testing {len(thresholds)} thresholds: {thresholds[0]} to {thresholds[-1]}")
    all_metrics = []

    for i, thresh in enumerate(thresholds):
        print(f"  [{i+1}/{len(thresholds)}] threshold={thresh:.3f} ...", end=" ", flush=True)
        t0 = time.time()

        metrics = evaluator.evaluate_dataset(threshold=thresh)
        elapsed = time.time() - t0

        row = {
            "threshold": thresh,
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "tn": metrics["tn"],
            "fn": metrics["fn"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "fpr": metrics["fpr"],
            "accuracy": metrics["accuracy"],
            "eval_time_s": round(elapsed, 2),
        }
        all_metrics.append(row)

        print(f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
              f"F1={metrics['f1']:.3f} FPR={metrics['fpr']:.3f} ({elapsed:.1f}s)")

    return all_metrics


def find_optimal_thresholds(metrics: List[Dict]) -> Dict:
    """
    Find optimal thresholds for different optimization targets.

    Returns dict with:
        - best_f1: threshold maximizing F1
        - best_balanced: threshold maximizing F1 with FPR < 0.05
        - best_recall: threshold maximizing recall with precision > 0.80
    """
    results = {}

    # Best F1
    best_f1 = max(metrics, key=lambda m: m["f1"])
    results["best_f1"] = {
        "threshold": best_f1["threshold"],
        "f1": best_f1["f1"],
        "precision": best_f1["precision"],
        "recall": best_f1["recall"],
        "fpr": best_f1["fpr"],
    }

    # Best balanced (F1 with FPR < 5%)
    low_fpr = [m for m in metrics if m["fpr"] < 0.05]
    if low_fpr:
        best_balanced = max(low_fpr, key=lambda m: m["f1"])
        results["best_balanced"] = {
            "threshold": best_balanced["threshold"],
            "f1": best_balanced["f1"],
            "precision": best_balanced["precision"],
            "recall": best_balanced["recall"],
            "fpr": best_balanced["fpr"],
        }

    # Best recall with precision > 80%
    high_prec = [m for m in metrics if m["precision"] > 0.80]
    if high_prec:
        best_recall = max(high_prec, key=lambda m: m["recall"])
        results["best_high_recall"] = {
            "threshold": best_recall["threshold"],
            "f1": best_recall["f1"],
            "precision": best_recall["precision"],
            "recall": best_recall["recall"],
            "fpr": best_recall["fpr"],
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Prompt Injection Threshold Optimizer")
    parser.add_argument("--dataset", type=str, default=str(_DATASET_PATH))
    parser.add_argument("--model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--min-thresh", type=float, default=0.30)
    parser.add_argument("--max-thresh", type=float, default=0.90)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Initialize evaluator (loads model + attack signatures once)
    evaluator = SemanticEvaluator(
        dataset_path=args.dataset,
        embedding_model=args.model,
        threshold=0.5,  # doesn't matter, we override per-eval
    )

    # Run optimization
    print(f"\n{'='*60}")
    print(f"  THRESHOLD OPTIMIZATION")
    print(f"  Range: {args.min_thresh} to {args.max_thresh}, step {args.step}")
    print(f"{'='*60}\n")

    metrics = run_threshold_optimization(
        evaluator,
        min_thresh=args.min_thresh,
        max_thresh=args.max_thresh,
        step=args.step,
    )

    # Save CSV
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else _OUTPUT_DIR / "week2_prompt_metrics.csv"

    fieldnames = ["threshold", "tp", "fp", "tn", "fn", "precision", "recall", "f1", "fpr", "accuracy", "eval_time_s"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    print(f"\n[Saved] {output_path}")

    # Find optimal thresholds
    optimal = find_optimal_thresholds(metrics)

    print(f"\n{'='*60}")
    print(f"  OPTIMAL THRESHOLDS")
    print(f"{'='*60}")

    if "best_f1" in optimal:
        o = optimal["best_f1"]
        print(f"\n  Best F1:")
        print(f"    Threshold: {o['threshold']}")
        print(f"    F1={o['f1']:.4f}  P={o['precision']:.4f}  R={o['recall']:.4f}  FPR={o['fpr']:.4f}")

    if "best_balanced" in optimal:
        o = optimal["best_balanced"]
        print(f"\n  Best Balanced (F1 with FPR < 5%):")
        print(f"    Threshold: {o['threshold']}")
        print(f"    F1={o['f1']:.4f}  P={o['precision']:.4f}  R={o['recall']:.4f}  FPR={o['fpr']:.4f}")

    if "best_high_recall" in optimal:
        o = optimal["best_high_recall"]
        print(f"\n  Best Recall (with Precision > 80%):")
        print(f"    Threshold: {o['threshold']}")
        print(f"    F1={o['f1']:.4f}  P={o['precision']:.4f}  R={o['recall']:.4f}  FPR={o['fpr']:.4f}")

    # Recommendation
    recommended = optimal.get("best_balanced", optimal.get("best_f1", {}))
    if recommended:
        print(f"\n  >>> RECOMMENDED THRESHOLD: {recommended.get('threshold', 'N/A')} <<<")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
