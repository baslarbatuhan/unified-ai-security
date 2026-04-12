"""
evaluation/tune_prompt_threshold.py
======================================
Adaptive Semantic Threshold Tuning.

Sweeps different threshold configurations for the prompt guard
semantic evaluator to find the optimal balance between detection
rate and false positive rate.

Tests:
    - Fixed thresholds: 0.50, 0.55, 0.60, 0.65, 0.70
    - Adaptive mode: short=0.55, medium=0.60, long=0.65
    - Aggressive adaptive: short=0.50, medium=0.55, long=0.60

Output:
    runs/prompt_threshold_tuning.csv

Usage:
    python evaluation/tune_prompt_threshold.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_RUNS_DIR = _PROJECT_ROOT / "runs"


# Threshold configurations to test
THRESHOLD_CONFIGS = [
    {"name": "fixed_0.50", "mode": "fixed", "threshold": 0.50},
    {"name": "fixed_0.55", "mode": "fixed", "threshold": 0.55},
    {"name": "fixed_0.60", "mode": "fixed", "threshold": 0.60},
    {"name": "fixed_0.65", "mode": "fixed", "threshold": 0.65},
    {"name": "fixed_0.70", "mode": "fixed", "threshold": 0.70},
    {"name": "adaptive_default", "mode": "adaptive", "short": 0.55, "medium": 0.60, "long": 0.65},
    {"name": "adaptive_aggressive", "mode": "adaptive", "short": 0.50, "medium": 0.55, "long": 0.60},
    {"name": "adaptive_conservative", "mode": "adaptive", "short": 0.60, "medium": 0.65, "long": 0.70},
]


def load_test_data() -> tuple:
    """Load injection and benign prompts for evaluation."""
    csv_path = _PROJECT_ROOT / "datasets" / "injection_prompts" / "injection_dataset_v1.csv"
    attacks = []
    benign = []

    if not csv_path.exists():
        print(f"  ERROR: {csv_path} not found")
        sys.exit(1)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("label", "0") == "1":
                attacks.append(row["prompt"])
            else:
                benign.append(row["prompt"])

    return attacks, benign


def evaluate_config(config: Dict, attacks: List[str], benign: List[str]) -> Dict:
    """Evaluate a threshold configuration."""
    from prompt_guard.pipeline import PromptGuardPipeline

    if config["mode"] == "fixed":
        pipeline = PromptGuardPipeline(semantic_threshold=config["threshold"])
    else:
        pipeline = PromptGuardPipeline(semantic_threshold=config["long"])

    tp, fp, tn, fn = 0, 0, 0, 0

    # Test attacks
    for prompt in attacks:
        result = pipeline.run(prompt)

        # For adaptive mode, manually override threshold behavior
        if config["mode"] == "adaptive":
            prompt_len = len(result.normalized_prompt)
            if prompt_len < 50:
                thresh = config["short"]
            elif prompt_len < 200:
                thresh = config["medium"]
            else:
                thresh = config["long"]
            # Re-check with adaptive threshold
            detected = (result.semantic and result.semantic.semantic_score >= thresh) or \
                       (result.pattern and result.pattern.is_detected)
        else:
            detected = result.is_injection

        if detected:
            tp += 1
        else:
            fn += 1

    # Test benign
    for prompt in benign:
        result = pipeline.run(prompt)

        if config["mode"] == "adaptive":
            prompt_len = len(result.normalized_prompt)
            if prompt_len < 50:
                thresh = config["short"]
            elif prompt_len < 200:
                thresh = config["medium"]
            else:
                thresh = config["long"]
            detected = (result.semantic and result.semantic.semantic_score >= thresh) or \
                       (result.pattern and result.pattern.is_detected)
        else:
            detected = result.is_injection

        if detected:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "config": config["name"],
        "mode": config["mode"],
        "total": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
    }


def run_tuning():
    """Run threshold tuning across all configurations."""
    print(f"\n{'='*65}")
    print(f"  PROMPT THRESHOLD TUNING")
    print(f"{'='*65}")

    attacks, benign = load_test_data()
    print(f"  Loaded {len(attacks)} attacks, {len(benign)} benign prompts\n")

    csv_rows = []
    for config in THRESHOLD_CONFIGS:
        print(f"  Testing {config['name']}...")
        t0 = time.time()
        metrics = evaluate_config(config, attacks, benign)
        elapsed = int((time.time() - t0) * 1000)
        metrics["elapsed_ms"] = elapsed
        csv_rows.append(metrics)
        print(f"    P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
              f"F1={metrics['f1']:.4f} FPR={metrics['fpr']:.4f} ({elapsed}ms)")

    # Save CSV
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RUNS_DIR / "prompt_threshold_tuning.csv"
    fieldnames = list(csv_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n  [Saved] {csv_path}")

    # Find best F1
    best = max(csv_rows, key=lambda r: r["f1"])
    print(f"\n  BEST: {best['config']} (F1={best['f1']:.4f}, FPR={best['fpr']:.4f})")
    print(f"{'='*65}")

    return csv_rows


if __name__ == "__main__":
    run_tuning()
