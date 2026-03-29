"""
tests/evaluation/prompt_injection_tests.py
=============================================
End-to-end prompt injection detection test suite.

Pipeline:
    1. Load injection_dataset_v1.csv
    2. For each prompt: run semantic evaluator
    3. Compute risk score
    4. Compare with ground truth label
    5. Output runs/prompt_metrics.csv

Metrics:
    - Precision, Recall, F1 Score
    - Per-category breakdown (if available)

Usage:
    python evaluation/prompt_injection_tests.py
    python evaluation/prompt_injection_tests.py --model BAAI/bge-large-en-v1.5
    python evaluation/prompt_injection_tests.py --threshold 0.60
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FILE_DIR.parent if _FILE_DIR.name == "evaluation" else _FILE_DIR
_DATASET_PATH = _PROJECT_ROOT / "datasets" / "injection_prompts" / "injection_dataset_v1.csv"
_PATTERN_LIB_PATH = _PROJECT_ROOT / "prompt_guard" / "pattern_library.json"
_RUNS_DIR = _PROJECT_ROOT / "runs"

sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from prompt_guard.semantic_evaluator_v1 import SemanticEvaluator
except ImportError:
    sys.path.insert(0, str(_PROJECT_ROOT / "prompt_guard"))
    from semantic_evaluator_v1 import SemanticEvaluator


# ---------------------------------------------------------------------------
# Pattern library checker (rule-based layer)
# ---------------------------------------------------------------------------
def load_pattern_library(path: Path = _PATTERN_LIB_PATH) -> List[Dict]:
    """Load patterns from pattern_library.json for rule-based pre-filtering."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    patterns = []
    for category, info in data.get("patterns", {}).items():
        for p in info.get("patterns", []):
            patterns.append({
                "id": p["id"],
                "category": category,
                "name": p["name"],
                "regex": p["regex"],
                "weight": p["weight"],
            })
    return patterns


def check_patterns(text: str, patterns: List[Dict]) -> List[str]:
    """Check text against pattern library, return matched pattern IDs."""
    import re
    matched = []
    for p in patterns:
        try:
            if re.search(p["regex"], text):
                matched.append(p["id"])
        except re.error:
            continue
    return matched


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
def run_prompt_injection_tests(
    dataset_path: Path = _DATASET_PATH,
    embedding_model: str = "BAAI/bge-m3",
    threshold: float = 0.65,
) -> Dict:
    """
    Run full prompt injection detection test pipeline.

    For each prompt in the dataset:
        1. Run semantic evaluator (cosine similarity)
        2. Run pattern library check (regex)
        3. Record decision, risk_score, matches
        4. Compare with ground truth label

    Returns dict with aggregated metrics.
    """
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return {}

    # Load dataset
    all_prompts = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_prompts.append({
                "prompt": row["prompt"].strip().strip('"'),
                "label": int(row["label"].strip().strip('"')),
            })

    print(f"\n{'='*65}")
    print(f"  PROMPT INJECTION TEST SUITE")
    print(f"  Model: {embedding_model}")
    print(f"  Threshold: {threshold}")
    print(f"  Dataset: {len(all_prompts)} prompts")
    print(f"{'='*65}")

    # Initialize evaluator
    evaluator = SemanticEvaluator(
        dataset_path=dataset_path,
        embedding_model=embedding_model,
        threshold=threshold,
    )

    # Load pattern library
    patterns = load_pattern_library()
    print(f"  Pattern library: {len(patterns)} patterns loaded")

    # Run evaluation with leave-one-out
    print(f"\n  Running evaluation...")
    eval_metrics = evaluator.evaluate_dataset(threshold=threshold)

    # Also run pattern checks on all prompts
    t0 = time.time()
    results = []
    pattern_tp = pattern_fp = pattern_tn = pattern_fn = 0

    for i, prompt_data in enumerate(all_prompts):
        prompt = prompt_data["prompt"]
        actual = prompt_data["label"]

        # Pattern check
        matched_patterns = check_patterns(prompt, patterns)
        pattern_detected = len(matched_patterns) > 0

        # Get semantic result from eval_metrics details
        if i < len(eval_metrics.get("details", [])):
            detail = eval_metrics["details"][i]
            semantic_score = detail["similarity"]
            semantic_detected = detail["predicted"] == 1
        else:
            semantic_score = 0.0
            semantic_detected = False

        # Combined: either semantic OR pattern detects
        combined_detected = semantic_detected or pattern_detected

        # Determine decision
        if semantic_score >= 0.85 or (semantic_detected and pattern_detected):
            decision = "block"
            risk_score = min(semantic_score * 1.05, 1.0)
        elif semantic_detected or (pattern_detected and semantic_score > 0.50):
            decision = "flag"
            risk_score = semantic_score
        elif pattern_detected:
            decision = "sanitize"
            risk_score = max(semantic_score, 0.40)
        else:
            decision = "allow"
            risk_score = semantic_score * 0.45

        # Pattern-only metrics
        if actual == 1 and pattern_detected:
            pattern_tp += 1
        elif actual == 0 and pattern_detected:
            pattern_fp += 1
        elif actual == 0 and not pattern_detected:
            pattern_tn += 1
        else:
            pattern_fn += 1

        results.append({
            "module": "prompt_guard",
            "test_case": prompt[:80],
            "decision": decision,
            "risk_score": round(risk_score, 4),
            "latency": 0,
            "actual_label": actual,
            "predicted_label": 1 if combined_detected else 0,
            "semantic_score": round(semantic_score, 4),
            "threshold": threshold,
            "pattern_matches": ",".join(matched_patterns) if matched_patterns else "",
            "detection_method": "both" if semantic_detected and pattern_detected
                               else "semantic" if semantic_detected
                               else "pattern" if pattern_detected
                               else "none",
        })

    eval_time = int((time.time() - t0) * 1000)

    # Combined metrics
    c_tp = sum(1 for r in results if r["actual_label"] == 1 and r["predicted_label"] == 1)
    c_fp = sum(1 for r in results if r["actual_label"] == 0 and r["predicted_label"] == 1)
    c_tn = sum(1 for r in results if r["actual_label"] == 0 and r["predicted_label"] == 0)
    c_fn = sum(1 for r in results if r["actual_label"] == 1 and r["predicted_label"] == 0)

    c_precision = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
    c_recall = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
    c_f1 = 2 * c_precision * c_recall / (c_precision + c_recall) if (c_precision + c_recall) > 0 else 0.0
    c_fpr = c_fp / (c_fp + c_tn) if (c_fp + c_tn) > 0 else 0.0

    # Semantic-only metrics (from evaluator)
    s_precision = eval_metrics.get("precision", 0)
    s_recall = eval_metrics.get("recall", 0)
    s_f1 = eval_metrics.get("f1", 0)
    s_fpr = eval_metrics.get("fpr", 0)

    # Pattern-only metrics
    p_precision = pattern_tp / (pattern_tp + pattern_fp) if (pattern_tp + pattern_fp) > 0 else 0.0
    p_recall = pattern_tp / (pattern_tp + pattern_fn) if (pattern_tp + pattern_fn) > 0 else 0.0
    p_f1 = 2 * p_precision * p_recall / (p_precision + p_recall) if (p_precision + p_recall) > 0 else 0.0

    # Print comparison table
    print(f"\n{'='*65}")
    print(f"  DETECTION METHOD COMPARISON (threshold={threshold})")
    print(f"{'='*65}")
    print(f"  {'Method':<18} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} {'Prec':>7} {'Recall':>7} {'F1':>7} {'FPR':>7}")
    print(f"  {'-'*18} {'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    print(f"  {'Semantic only':<18} {eval_metrics.get('tp',0):>4} {eval_metrics.get('fp',0):>4} {eval_metrics.get('tn',0):>4} {eval_metrics.get('fn',0):>4} {s_precision:>7.3f} {s_recall:>7.3f} {s_f1:>7.3f} {s_fpr:>7.3f}")
    print(f"  {'Pattern only':<18} {pattern_tp:>4} {pattern_fp:>4} {pattern_tn:>4} {pattern_fn:>4} {p_precision:>7.3f} {p_recall:>7.3f} {p_f1:>7.3f} {pattern_fp/(pattern_fp+pattern_tn) if (pattern_fp+pattern_tn)>0 else 0:>7.3f}")
    print(f"  {'Combined':<18} {c_tp:>4} {c_fp:>4} {c_tn:>4} {c_fn:>4} {c_precision:>7.3f} {c_recall:>7.3f} {c_f1:>7.3f} {c_fpr:>7.3f}")

    # Detection method breakdown
    method_counts = {}
    for r in results:
        if r["predicted_label"] == 1:
            m = r["detection_method"]
            method_counts[m] = method_counts.get(m, 0) + 1

    if method_counts:
        print(f"\n  Detection method breakdown (for detected attacks):")
        for m, c in sorted(method_counts.items(), key=lambda x: -x[1]):
            print(f"    {m}: {c}")

    # False negatives
    fn_list = [r for r in results if r["actual_label"] == 1 and r["predicted_label"] == 0]
    if fn_list:
        print(f"\n  FALSE NEGATIVES ({len(fn_list)} missed attacks):")
        for r in fn_list[:10]:
            print(f"    sem={r['semantic_score']:.3f} | {r['test_case']}")

    # False positives
    fp_list = [r for r in results if r["actual_label"] == 0 and r["predicted_label"] == 1]
    if fp_list:
        print(f"\n  FALSE POSITIVES ({len(fp_list)} wrongly flagged):")
        for r in fp_list[:10]:
            method = r["detection_method"]
            print(f"    sem={r['semantic_score']:.3f} | via={method} | {r['test_case']}")

    print(f"{'='*65}")

    # ---- SAVE CSV ----
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RUNS_DIR / "prompt_metrics.csv"
    fieldnames = [
        "module", "test_case", "decision", "risk_score", "latency",
        "actual_label", "predicted_label", "semantic_score", "threshold",
        "pattern_matches", "detection_method",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  [Saved] {csv_path}")

    return {
        "model": embedding_model,
        "threshold": threshold,
        "total_prompts": len(all_prompts),
        "semantic": {"tp": eval_metrics.get("tp",0), "fp": eval_metrics.get("fp",0),
                     "tn": eval_metrics.get("tn",0), "fn": eval_metrics.get("fn",0),
                     "precision": s_precision, "recall": s_recall, "f1": s_f1, "fpr": s_fpr},
        "pattern": {"tp": pattern_tp, "fp": pattern_fp, "tn": pattern_tn, "fn": pattern_fn,
                    "precision": round(p_precision, 4), "recall": round(p_recall, 4), "f1": round(p_f1, 4)},
        "combined": {"tp": c_tp, "fp": c_fp, "tn": c_tn, "fn": c_fn,
                     "precision": round(c_precision, 4), "recall": round(c_recall, 4),
                     "f1": round(c_f1, 4), "fpr": round(c_fpr, 4)},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prompt Injection Detection Tests")
    parser.add_argument("--dataset", type=str, default=str(_DATASET_PATH))
    parser.add_argument("--model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()

    metrics = run_prompt_injection_tests(
        dataset_path=Path(args.dataset),
        embedding_model=args.model,
        threshold=args.threshold,
    )

    if metrics:
        summary_path = _RUNS_DIR / "prompt_test_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"  [Saved] {summary_path}")


if __name__ == "__main__":
    main()
