"""
evaluation/fusion_threshold_optimization.py
==============================================
Fusion katmanı threshold optimizasyonu.

Farklı threshold kombinasyonları test ederek en iyi
güvenlik-performans dengesi bulunur.

Tested ranges:
    allow:    0.15 - 0.40
    sanitize: 0.45 - 0.70
    block:    0.75 - 0.95

Çıktı:
    runs/fusion_threshold_analysis.csv

Usage:
    python evaluation/fusion_threshold_optimization.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

_FILE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FILE_DIR.parent if _FILE_DIR.name == "evaluation" else _FILE_DIR
_RUNS_DIR = _PROJECT_ROOT / "runs"
sys.path.insert(0, str(_PROJECT_ROOT))

from fusion_gateway.engine import FusionEngine


def _decide(score: float, allow_th: float, sanitize_th: float, block_th: float) -> str:
    if score < allow_th:
        return "allow"
    elif score < sanitize_th:
        return "sanitize"
    elif score >= block_th:
        return "block"
    else:
        return "flag"


def load_test_cases() -> List[Dict]:
    """Load a mix of attack and benign cases for threshold testing."""
    cases = []

    # Prompt injection attacks
    csv_path = _PROJECT_ROOT / "datasets" / "injection_prompts" / "injection_dataset_v1.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cases.append({
                    "user_input": row["prompt"],
                    "is_attack": row.get("label", "0") == "1",
                    "source": "prompt",
                })

    # RAG poisoned docs
    json_path = _PROJECT_ROOT / "datasets" / "poisoned_corpus" / "poison_samples.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        for doc in dataset["documents"]:
            cases.append({
                "user_input": doc.get("target_query", "General query"),
                "retrieved_context": doc["content"],
                "is_attack": doc.get("is_poisoned", False),
                "source": "rag",
            })

    return cases


def run_threshold_optimization() -> Dict:
    """Test multiple threshold combinations."""

    print(f"\n{'='*65}")
    print(f"  FUSION THRESHOLD OPTIMIZATION")
    print(f"{'='*65}")

    # Load test cases
    cases = load_test_cases()
    print(f"  Test cases: {len(cases)} ({sum(1 for c in cases if c['is_attack'])} attacks, "
          f"{sum(1 for c in cases if not c['is_attack'])} benign)")

    # First: compute fused scores for all cases (expensive, do once)
    engine = FusionEngine()
    print(f"\n  Computing fused scores for {len(cases)} cases...")

    scored_cases = []
    for i, case in enumerate(cases):
        response = engine.analyze(
            user_input=case.get("user_input", ""),
            retrieved_context=case.get("retrieved_context"),
        )
        scored_cases.append({
            "fused_risk": response.fused_risk,
            "is_attack": case["is_attack"],
            "source": case["source"],
        })
        if (i + 1) % 50 == 0:
            print(f"    Scored {i+1}/{len(cases)}...")

    print(f"    Done. Scored {len(scored_cases)} cases.")

    # Threshold combinations to test
    allow_range = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    sanitize_range = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    block_range = [0.75, 0.80, 0.85, 0.90, 0.95]

    results = []
    best_f1 = 0
    best_config = None

    print(f"\n  Testing {len(allow_range) * len(sanitize_range) * len(block_range)} threshold combinations...")

    for allow_th in allow_range:
        for sanitize_th in sanitize_range:
            for block_th in block_range:
                if allow_th >= sanitize_th or sanitize_th >= block_th:
                    continue

                tp = fp = tn = fn = 0
                decisions = {"allow": 0, "sanitize": 0, "flag": 0, "block": 0}

                for sc in scored_cases:
                    decision = _decide(sc["fused_risk"], allow_th, sanitize_th, block_th)
                    decisions[decision] += 1

                    is_blocked = decision in ("block", "flag", "sanitize")

                    if sc["is_attack"] and is_blocked:
                        tp += 1
                    elif sc["is_attack"] and not is_blocked:
                        fn += 1
                    elif not sc["is_attack"] and is_blocked:
                        fp += 1
                    else:
                        tn += 1

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                row = {
                    "allow_threshold": allow_th,
                    "sanitize_threshold": sanitize_th,
                    "block_threshold": block_th,
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "fpr": round(fpr, 4),
                    "allow_count": decisions["allow"],
                    "sanitize_count": decisions["sanitize"],
                    "flag_count": decisions["flag"],
                    "block_count": decisions["block"],
                }
                results.append(row)

                if f1 > best_f1:
                    best_f1 = f1
                    best_config = row

    # Sort by F1
    results.sort(key=lambda x: x["f1"], reverse=True)

    # Print top 10
    print(f"\n  TOP 10 CONFIGURATIONS:")
    print(f"  {'Allow':>6s} {'Sanit':>6s} {'Block':>6s} | {'Prec':>6s} {'Recall':>6s} {'F1':>6s} {'FPR':>6s} | {'TP':>4s} {'FP':>4s} {'FN':>4s}")
    print(f"  {'-'*6} {'-'*6} {'-'*6} | {'-'*6} {'-'*6} {'-'*6} {'-'*6} | {'-'*4} {'-'*4} {'-'*4}")
    for r in results[:10]:
        print(f"  {r['allow_threshold']:>6.2f} {r['sanitize_threshold']:>6.2f} {r['block_threshold']:>6.2f} | "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} {r['fpr']:>6.3f} | "
              f"{r['tp']:>4d} {r['fp']:>4d} {r['fn']:>4d}")

    # Current config
    print(f"\n  CURRENT CONFIG (secure_balanced.yaml): allow=0.30, sanitize=0.60, block=0.85")
    current = [r for r in results if r["allow_threshold"] == 0.30
               and r["sanitize_threshold"] == 0.60 and r["block_threshold"] == 0.85]
    if current:
        c = current[0]
        print(f"    Precision={c['precision']:.3f} Recall={c['recall']:.3f} F1={c['f1']:.3f} FPR={c['fpr']:.3f}")

    if best_config:
        print(f"\n  BEST CONFIG: allow={best_config['allow_threshold']}, "
              f"sanitize={best_config['sanitize_threshold']}, block={best_config['block_threshold']}")
        print(f"    F1={best_config['f1']:.3f} Precision={best_config['precision']:.3f} "
              f"Recall={best_config['recall']:.3f} FPR={best_config['fpr']:.3f}")

    # Save CSV
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RUNS_DIR / "fusion_threshold_analysis.csv"
    fieldnames = list(results[0].keys()) if results else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  [Saved] {csv_path} ({len(results)} combinations)")

    print(f"\n{'='*65}")
    return {"total_combinations": len(results), "best_f1": best_f1, "best_config": best_config}


if __name__ == "__main__":
    run_threshold_optimization()
