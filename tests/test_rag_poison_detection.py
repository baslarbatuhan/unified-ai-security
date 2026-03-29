"""
tests/test_rag_poison_detection.py
====================================
End-to-end RAG poison detection test suite.

Pipeline:
    1. Load poison_samples.json
    2. For each target query: retrieve documents
    3. Run poison_detector on retrieved docs
    4. Compute risk_score
    5. Compare with ground truth
    6. Output runs/rag_metrics.csv

Validates that poisoned documents are detected within the retrieval pipeline.

Usage:
    python tests/test_rag_poison_detection.py
    python tests/test_rag_poison_detection.py --model BAAI/bge-large-en-v1.5
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
_PROJECT_ROOT = _FILE_DIR.parent if _FILE_DIR.name == "tests" else _FILE_DIR
_DATASET_PATH = _PROJECT_ROOT / "datasets" / "poisoned_corpus" / "poison_samples.json"
_RUNS_DIR = _PROJECT_ROOT / "runs"

sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from rag_guard.poison_detector import PoisonDetector
    from rag_guard.risk_scoring import RAGRiskScorer, RiskThresholds
except ImportError:
    sys.path.insert(0, str(_PROJECT_ROOT / "rag_guard"))
    from poison_detector import PoisonDetector
    from risk_scoring import RAGRiskScorer, RiskThresholds


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
def run_rag_poison_tests(
    dataset_path: Path = _DATASET_PATH,
    embedding_model: str = "BAAI/bge-m3",
) -> Dict:
    """
    Run full RAG poison detection test pipeline.

    Steps per poisoned document's target query:
        1. Simulate retrieval (poisoned doc + clean docs as top-k)
        2. Run poison_detector
        3. Compute risk_score
        4. Compare detector output with ground truth

    Returns dict with metrics and per-test results.
    """
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    documents = dataset["documents"]
    clean_docs = [d for d in documents if not d.get("is_poisoned")]
    poisoned_docs = [d for d in documents if d.get("is_poisoned")]

    # Initialize detector and scorer
    detector = PoisonDetector(embedding_model=embedding_model)
    scorer = RAGRiskScorer()

    print(f"\n{'='*65}")
    print(f"  RAG POISON DETECTION TEST SUITE")
    print(f"  Model: {embedding_model}")
    print(f"  Documents: {len(clean_docs)} clean + {len(poisoned_docs)} poisoned")
    print(f"{'='*65}")

    # ---- TEST SET 1: Poisoned target queries ----
    # Each poisoned doc has a target_query — simulate retrieval with that doc in top-5
    tp = fp = tn = fn = 0
    results = []
    total_latency = 0

    print(f"\n  [TEST SET 1] Target queries with poisoned documents")

    for pdoc in poisoned_docs:
        target_query = pdoc.get("target_query")
        if not target_query:
            continue

        # Simulate retrieval: this poisoned doc + 4 clean docs
        simulated_retrieval = [
            {"doc_id": pdoc["doc_id"], "content": pdoc["content"]},
        ] + [
            {"doc_id": d["doc_id"], "content": d["content"]}
            for d in clean_docs[:4]
        ]

        t0 = time.time()

        # Step 3: Run poison detector
        detection = detector.detect(simulated_retrieval)

        # Step 4: Compute risk score
        risk = scorer.score(detection)

        latency = int((time.time() - t0) * 1000)
        total_latency += latency

        # Step 5: Check if poisoned doc was detected
        poisoned_doc_score = detection.document_scores[0]  # first doc is the poisoned one
        detected = poisoned_doc_score.is_suspicious

        if detected:
            tp += 1
        else:
            fn += 1

        status = "TP" if detected else "FN"
        print(f"    [{status}] {pdoc['doc_id']:12s} | score={poisoned_doc_score.poison_score:.3f} | "
              f"risk={risk.risk_score:.3f} | decision={risk.decision} | "
              f"type={pdoc['poison_type']}")

        results.append({
            "module": "rag_guard",
            "test_case": target_query[:80],
            "decision": risk.decision,
            "risk_score": round(risk.risk_score, 4),
            "latency": latency,
            "query": target_query[:80],
            "poisoned_count": detection.suspicious_count,
            "total_retrieved": detection.total_documents,
            "poison_ratio": round(detection.suspicion_ratio, 4),
            "actual_label": 1,
            "predicted_label": 1 if detected else 0,
            "poison_type": pdoc["poison_type"],
            "doc_id": pdoc["doc_id"],
        })

    # ---- TEST SET 2: Clean-only retrieval (no poisoned docs) ----
    print(f"\n  [TEST SET 2] Clean-only retrieval (false positive test)")

    clean_queries = [
        "How do I deploy a machine learning model?",
        "What is the best branching strategy for Git?",
        "Explain microservices architecture",
        "How does Kubernetes handle scaling?",
        "What monitoring tools should I use?",
        "How do I set up CI/CD pipelines?",
        "What is the difference between REST and GraphQL?",
        "Explain the SOLID principles in OOP.",
        "How do I handle database migrations?",
        "What is containerization?",
    ]

    for query in clean_queries:
        # Only clean docs in retrieval
        simulated_retrieval = [
            {"doc_id": d["doc_id"], "content": d["content"]}
            for d in clean_docs[:5]
        ]

        t0 = time.time()
        detection = detector.detect(simulated_retrieval)
        risk = scorer.score(detection)
        latency = int((time.time() - t0) * 1000)
        total_latency += latency

        false_alarm = detection.suspicious_count > 0
        if false_alarm:
            fp += 1
            status = "FP"
        else:
            tn += 1
            status = "TN"

        print(f"    [{status}] suspicious={detection.suspicious_count} | "
              f"risk={risk.risk_score:.3f} | decision={risk.decision} | "
              f"\"{query[:50]}\"")

        results.append({
            "module": "rag_guard",
            "test_case": query[:80],
            "decision": risk.decision,
            "risk_score": round(risk.risk_score, 4),
            "latency": latency,
            "query": query[:80],
            "poisoned_count": detection.suspicious_count,
            "total_retrieved": detection.total_documents,
            "poison_ratio": round(detection.suspicion_ratio, 4),
            "actual_label": 0,
            "predicted_label": 1 if false_alarm else 0,
            "poison_type": "none",
            "doc_id": "clean_set",
        })

    # ---- METRICS ----
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    total_tests = tp + fp + tn + fn
    accuracy = (tp + tn) / total_tests if total_tests > 0 else 0.0

    print(f"\n{'='*65}")
    print(f"  RAG POISON DETECTION METRICS")
    print(f"{'='*65}")
    print(f"  TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1 Score:   {f1:.4f}")
    print(f"  FPR:        {fpr:.4f}")
    print(f"  Accuracy:   {accuracy:.4f}")
    print(f"  Avg latency: {total_latency / total_tests:.0f}ms per query")

    # Missed attacks breakdown
    fn_results = [r for r in results if r["actual_label"] == 1 and r["predicted_label"] == 0]
    if fn_results:
        print(f"\n  FALSE NEGATIVES ({len(fn_results)} missed):")
        for r in fn_results:
            print(f"    {r['doc_id']} | {r['poison_type']} | \"{r['query'][:50]}\"")

    print(f"{'='*65}")

    # ---- SAVE CSV (metrics_schema format) ----
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RUNS_DIR / "rag_metrics.csv"
    fieldnames = [
        "module", "test_case", "decision", "risk_score", "latency",
        "query", "poisoned_count", "total_retrieved", "poison_ratio",
        "actual_label", "predicted_label", "poison_type", "doc_id",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n  [Saved] {csv_path}")

    return {
        "model": embedding_model,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "accuracy": round(accuracy, 4),
        "total_tests": total_tests,
        "avg_latency_ms": round(total_latency / total_tests) if total_tests > 0 else 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RAG Poison Detection Tests")
    parser.add_argument("--dataset", type=str, default=str(_DATASET_PATH))
    parser.add_argument("--model", type=str, default="BAAI/bge-m3")
    args = parser.parse_args()

    metrics = run_rag_poison_tests(
        dataset_path=Path(args.dataset),
        embedding_model=args.model,
    )

    if metrics:
        # Save summary JSON
        summary_path = _RUNS_DIR / "rag_test_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"  [Saved] {summary_path}")


if __name__ == "__main__":
    main()
