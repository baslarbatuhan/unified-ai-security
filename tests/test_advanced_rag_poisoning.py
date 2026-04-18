"""
tests/test_advanced_rag_poisoning.py
=======================================
Advanced RAG poisoning evasion tests.

Pipeline:
    advanced_poison_samples.json → poison_detector → context_filter → retrieval_risk_score

Metrics: top_k_poison_ratio, poison_detection_recall, attack_success_rate, poison_doc_ids

Usage:
    python tests/test_advanced_rag_poisoning.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict

_FILE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FILE_DIR.parent if _FILE_DIR.name == "tests" else _FILE_DIR
_RUNS_DIR = _PROJECT_ROOT / "runs"
sys.path.insert(0, str(_PROJECT_ROOT))

# Load environment variables (HF_TOKEN, etc.) before importing any model code
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

try:
    from rag_guard.poison_detector import PoisonDetector
    from rag_guard.context_filter import ContextFilter
    from rag_guard.retrieval_risk_score import RetrievalRiskScorer, DocScore
except ImportError:
    sys.path.insert(0, str(_PROJECT_ROOT / "rag_guard"))
    from poison_detector import PoisonDetector
    from context_filter import ContextFilter
    from retrieval_risk_score import RetrievalRiskScorer, DocScore


def run_advanced_rag_tests() -> Dict:
    dataset_path = _PROJECT_ROOT / "datasets" / "poisoned_corpus" / "advanced_poison_samples.json"
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        return {"status": "error", "error": "dataset_not_found"}

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    documents = dataset["documents"]
    clean_docs = [d for d in documents if not d["is_poisoned"]]
    poisoned_docs = [d for d in documents if d["is_poisoned"]]

    print(f"\n{'='*65}")
    print(f"  ADVANCED RAG POISONING EVASION TESTS")
    print(f"  Documents: {len(clean_docs)} clean + {len(poisoned_docs)} poisoned")
    print(f"{'='*65}")

    detector = PoisonDetector()
    ctx_filter = ContextFilter(detector)
    risk_scorer = RetrievalRiskScorer()

    results = []
    tp, fp, tn, fn = 0, 0, 0, 0
    technique_results: Dict[str, Dict] = {}
    total_latency = 0

    # --- TEST SET 1: Poisoned document detection ---
    print(f"\n  [TEST SET 1] Poisoned document detection ({len(poisoned_docs)} docs)")

    for doc in poisoned_docs:
        t0 = time.time()
        detection = detector.detect([{"doc_id": doc["doc_id"], "content": doc["content"]}])
        latency_ms = int((time.time() - t0) * 1000)
        total_latency += latency_ms

        doc_score = detection.document_scores[0] if detection.document_scores else None
        score = doc_score.poison_score if doc_score else 0.0
        detected = score >= 0.55

        technique = doc.get("poison_technique", "unknown")
        if technique not in technique_results:
            technique_results[technique] = {"total": 0, "detected": 0, "evaded": 0, "scores": []}
        technique_results[technique]["total"] += 1
        technique_results[technique]["scores"].append(score)

        if detected:
            tp += 1
            technique_results[technique]["detected"] += 1
            status = "TP"
            decision = "sanitize"
        else:
            fn += 1
            technique_results[technique]["evaded"] += 1
            status = "FN"
            decision = "allow"

        risk_result = risk_scorer.score([DocScore(doc_id=doc["doc_id"], poison_score=score, rank=1)])

        print(f"    [{status}] {doc['doc_id']:18s} | score={score:.3f} | risk={risk_result.risk_score:.3f} | "
              f"technique={technique}")

        results.append({
            "module": "rag_guard",
            "test_case": doc["doc_id"],
            "decision": decision,
            "risk_score": round(risk_result.risk_score, 4),
            "latency": latency_ms,
            "poison_score": round(score, 4),
            "is_poisoned": True,
            "detected": detected,
            "poison_type": doc.get("poison_type", ""),
            "poison_technique": technique,
            "evasion_technique": doc.get("evasion_technique", ""),
            "test_set": "poisoned",
        })

    # --- TEST SET 2: Clean documents FP test ---
    print(f"\n  [TEST SET 2] Clean document FP test ({len(clean_docs)} docs)")

    for doc in clean_docs:
        t0 = time.time()
        detection = detector.detect([{"doc_id": doc["doc_id"], "content": doc["content"]}])
        latency_ms = int((time.time() - t0) * 1000)
        total_latency += latency_ms

        doc_score = detection.document_scores[0] if detection.document_scores else None
        score = doc_score.poison_score if doc_score else 0.0
        flagged = score >= 0.55

        if flagged:
            fp += 1
            status = "FP"
        else:
            tn += 1
            status = "TN"

        print(f"    [{status}] {doc['doc_id']:18s} | score={score:.3f}")

        results.append({
            "module": "rag_guard",
            "test_case": doc["doc_id"],
            "decision": "sanitize" if flagged else "allow",
            "risk_score": round(score * 0.3, 4),
            "latency": latency_ms,
            "poison_score": round(score, 4),
            "is_poisoned": False,
            "detected": flagged,
            "poison_type": "",
            "poison_technique": "",
            "evasion_technique": "",
            "test_set": "clean",
        })

    # --- TEST SET 3: Context filtering ---
    print(f"\n  [TEST SET 3] Context filtering (mixed 5 clean + 5 poisoned)")
    mixed = clean_docs[:5] + poisoned_docs[:5]
    filter_result = ctx_filter.sanitize(mixed)
    print(f"    Original: {filter_result.original_count} | Kept: {filter_result.kept_count} | "
          f"Removed: {filter_result.removed_count}")
    print(f"    Sanitization ratio: {filter_result.sanitization_ratio:.0%} | Needs review: {filter_result.needs_review}")

    # --- METRICS ---
    total_tests = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    evasion_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
    avg_latency = total_latency // total_tests if total_tests > 0 else 0

    print(f"\n{'='*65}")
    print(f"  ADVANCED RAG POISONING METRICS")
    print(f"{'='*65}")
    print(f"  TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")
    print(f"  Precision:       {precision:.4f}")
    print(f"  Recall:          {recall:.4f}")
    print(f"  F1 Score:        {f1:.4f}")
    print(f"  FPR:             {fpr:.4f}")
    print(f"  Evasion rate:    {evasion_rate:.4f} ({fn}/{tp+fn} evaded)")
    print(f"  Avg latency:     {avg_latency}ms")

    # Per-technique breakdown
    print(f"\n  EVASION BY TECHNIQUE:")
    print(f"  {'Technique':<25s} {'Total':>5s} {'Det':>5s} {'Evd':>5s} {'Evade%':>7s} {'AvgScore':>9s}")
    print(f"  {'-'*25} {'-'*5} {'-'*5} {'-'*5} {'-'*7} {'-'*9}")
    for tech, stats in sorted(technique_results.items()):
        evade_pct = stats["evaded"] / stats["total"] * 100 if stats["total"] > 0 else 0
        avg_sc = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
        print(f"  {tech:<25s} {stats['total']:>5d} {stats['detected']:>5d} {stats['evaded']:>5d} "
              f"{evade_pct:>6.1f}% {avg_sc:>9.3f}")

    fn_docs = [r for r in results if r["is_poisoned"] and not r["detected"]]
    if fn_docs:
        print(f"\n  EVADED ATTACKS ({len(fn_docs)}):")
        for r in fn_docs:
            print(f"    {r['test_case']} | {r['poison_technique']} | score={r['poison_score']:.3f}")

    print(f"{'='*65}")

    # --- SAVE ---
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RUNS_DIR / "rag_poison_metrics.csv"
    fieldnames = ["module","test_case","decision","risk_score","latency",
                  "poison_score","is_poisoned","detected","poison_type",
                  "poison_technique","evasion_technique","test_set"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  [Saved] {csv_path}")

    summary = {
        "dataset": "advanced_poison_samples.json",
        "total_documents": len(documents),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "fpr": round(fpr, 4),
        "evasion_rate": round(evasion_rate, 4),
        "avg_latency_ms": avg_latency,
        "technique_breakdown": {
            tech: {"total": s["total"], "detected": s["detected"], "evaded": s["evaded"],
                   "evasion_rate": round(s["evaded"]/s["total"], 4) if s["total"]>0 else 0,
                   "avg_score": round(sum(s["scores"])/len(s["scores"]), 4) if s["scores"] else 0}
            for tech, s in technique_results.items()
        },
        "context_filter": {
            "original": filter_result.original_count, "kept": filter_result.kept_count,
            "removed": filter_result.removed_count, "ratio": filter_result.sanitization_ratio,
        },
    }
    summary_path = _RUNS_DIR / "rag_advanced_test_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  [Saved] {summary_path}")

    return summary


if __name__ == "__main__":
    run_advanced_rag_tests()
