"""
evaluation/rag_weight_optimization.py
=========================================
RAG Guard Hybrid Weight Optimization.

Tests different embedding/LLM judge weight combinations to find
the optimal balance for poison detection.

Variants:
    - 40/60 (current)
    - 30/70
    - 20/80

Output:
    runs/rag_weight_analysis.csv

Usage:
    python evaluation/rag_weight_optimization.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_RUNS_DIR = _PROJECT_ROOT / "runs"


WEIGHT_VARIANTS = [
    {"name": "40_60", "embedding": 0.4, "judge": 0.6},
    {"name": "30_70", "embedding": 0.3, "judge": 0.7},
    {"name": "20_80", "embedding": 0.2, "judge": 0.8},
]


def load_dataset() -> Dict:
    """Load advanced poison samples."""
    path = _PROJECT_ROOT / "datasets" / "poisoned_corpus" / "advanced_poison_samples.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_variant(documents: List[Dict], labels: Dict[str, bool],
                embedding_weight: float, judge_weight: float) -> Dict:
    """Run RAG guard pipeline with specific weights and measure performance."""
    from rag_guard.pipeline import RAGGuardPipeline
    from rag_guard.llm_judge import LLMJudge

    judge = LLMJudge()
    pipeline = RAGGuardPipeline(
        judge=judge,
        embedding_weight=embedding_weight,
        judge_weight=judge_weight,
    )

    tp, fp, tn, fn = 0, 0, 0, 0
    t0 = time.time()

    result = pipeline.run(documents, user_query="What is the security policy?")

    for doc_score in result.doc_scores:
        doc_id = doc_score.doc_id
        is_poisoned = labels.get(doc_id, False)
        detected = doc_score.is_suspicious

        if is_poisoned and detected:
            tp += 1
        elif is_poisoned and not detected:
            fn += 1
        elif not is_poisoned and detected:
            fp += 1
        else:
            tn += 1

    elapsed = int((time.time() - t0) * 1000)
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "total": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "latency_ms": elapsed,
    }


def run_optimization():
    """Run all weight variants and compare."""
    print(f"\n{'='*65}")
    print(f"  RAG WEIGHT OPTIMIZATION")
    print(f"{'='*65}")

    dataset = load_dataset()
    documents = dataset["documents"]
    labels = {doc["doc_id"]: doc.get("is_poisoned", False) for doc in documents}

    # Prepare doc list for pipeline (needs doc_id + content)
    doc_list = [{"doc_id": d["doc_id"], "content": d["content"]} for d in documents]

    csv_rows = []
    for variant in WEIGHT_VARIANTS:
        print(f"\n  Testing {variant['name']} (emb={variant['embedding']}, judge={variant['judge']})...")
        metrics = run_variant(doc_list, labels, variant["embedding"], variant["judge"])

        print(f"    P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f} FPR={metrics['fpr']:.4f}")
        print(f"    TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")
        print(f"    Latency: {metrics['latency_ms']}ms")

        csv_rows.append({
            "variant": variant["name"],
            "embedding_weight": variant["embedding"],
            "judge_weight": variant["judge"],
            **metrics,
        })

    # Save CSV
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RUNS_DIR / "rag_weight_analysis.csv"
    fieldnames = list(csv_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n  [Saved] {csv_path}")

    # Find best
    best = max(csv_rows, key=lambda r: r["f1"])
    print(f"\n  BEST: {best['variant']} (F1={best['f1']:.4f})")
    print(f"{'='*65}")

    return csv_rows


if __name__ == "__main__":
    run_optimization()
