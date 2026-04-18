"""
tests/test_advanced_rag_hybrid.py
==================================
Advanced RAG poisoning test with FULL hybrid pipeline (embedding + LLM judge).

Compares against embedding-only baseline from test_advanced_rag_poisoning.py.

Pipeline: RAGGuardPipeline.run(use_judge=True)
  Stage 1: Embedding detection (PoisonDetector)
  Stage 2: LLM judge (qwen2.5:7b via Ollama)
  Stage 3: Score combination (weighted + abstention/floor rules)
  Stage 4: Risk scoring (RetrievalRiskScorer + judge amplification)
  Stage 5: Context filtering

Usage:
    python tests/test_advanced_rag_hybrid.py
    python tests/test_advanced_rag_hybrid.py --skip-judge   # embedding-only comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

_FILE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FILE_DIR.parent if _FILE_DIR.name == "tests" else _FILE_DIR
_RUNS_DIR = _PROJECT_ROOT / "runs"
_DATASET_PATH = _PROJECT_ROOT / "datasets" / "poisoned_corpus" / "advanced_poison_samples.json"

sys.path.insert(0, str(_PROJECT_ROOT))

# Load environment variables (HF_TOKEN, etc.) before importing any model code
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass


def run_advanced_hybrid_test(use_judge: bool = True) -> Dict:
    """Run advanced poison dataset through full RAGGuardPipeline."""

    from fusion_gateway.engine import _get_rag_pipeline, _load_yaml_config

    if not _DATASET_PATH.exists():
        print(f"ERROR: Dataset not found: {_DATASET_PATH}")
        return {}

    with open(_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    documents = dataset["documents"]
    clean_docs = [d for d in documents if not d["is_poisoned"]]
    poisoned_docs = [d for d in documents if d["is_poisoned"]]

    # Load config for display
    cfg = _load_yaml_config()
    rag_cfg = cfg.get("modules", {}).get("rag_guard", {})
    lj_cfg = rag_cfg.get("llm_judge", {})

    mode_str = "hybrid (embedding + LLM judge)" if use_judge else "embedding-only"

    print(f"\n{'='*70}")
    print(f"  ADVANCED RAG POISONING — {mode_str.upper()}")
    print(f"{'='*70}")
    print(f"  Dataset: advanced_poison_samples.json")
    print(f"  Documents: {len(clean_docs)} clean + {len(poisoned_docs)} poisoned")
    print(f"  Config: emb_w={lj_cfg.get('embedding_weight', 0.5)}, "
          f"judge_w={lj_cfg.get('judge_weight', 0.5)}, "
          f"poison_th={rag_cfg.get('poison_threshold', 0.45)}")
    print(f"  Judge abstain: {lj_cfg.get('judge_abstain_threshold', 0.15)}, "
          f"emb override mul: {lj_cfg.get('embedding_override_multiplier', 0.85)}")
    print(f"{'='*70}")

    pipeline = _get_rag_pipeline()

    tp = fp = tn = fn = 0
    results: List[Dict] = []
    technique_results: Dict[str, Dict] = {}
    total_latency = 0

    # --- TEST SET 1: Poisoned docs with target queries ---
    print(f"\n  [TEST SET 1] Poisoned document detection ({len(poisoned_docs)} attacks)")
    print(f"  Each attack: 1 poisoned + 4 clean padding docs\n")

    for pdoc in poisoned_docs:
        target_query = pdoc.get("target_query", "general query")
        technique = pdoc.get("poison_technique", "unknown")

        # Simulate top-5 retrieval: poisoned doc at rank 1, 4 clean padding
        simulated_retrieval = [
            {"doc_id": pdoc["doc_id"], "content": pdoc["content"]},
        ] + [
            {"doc_id": d["doc_id"], "content": d["content"]}
            for d in clean_docs[:4]
        ]

        t0 = time.time()
        result = pipeline.run(simulated_retrieval, user_query=target_query, use_judge=use_judge)
        latency_ms = int((time.time() - t0) * 1000)
        total_latency += latency_ms

        # Get poisoned doc's scores (first in retrieval)
        ds0 = result.doc_scores[0] if result.doc_scores else None
        emb_score = ds0.embedding_score if ds0 else 0.0
        judge_score = ds0.judge_score if ds0 else 0.0
        combined = ds0.combined_score if ds0 else 0.0
        detected = bool(ds0 and ds0.is_suspicious)
        explanation = ds0.judge_explanation if ds0 else ""

        rr = result.risk_result
        risk_score = rr.risk_score if rr else 0.0
        decision = rr.decision if rr else "allow"

        # Track technique stats
        if technique not in technique_results:
            technique_results[technique] = {
                "total": 0, "detected": 0, "evaded": 0,
                "emb_scores": [], "judge_scores": [], "combined_scores": [],
            }
        tr = technique_results[technique]
        tr["total"] += 1
        tr["emb_scores"].append(emb_score)
        tr["judge_scores"].append(judge_score)
        tr["combined_scores"].append(combined)

        if detected:
            tp += 1
            tr["detected"] += 1
            status = "TP"
        else:
            fn += 1
            tr["evaded"] += 1
            status = "FN"

        print(f"  [{status}] {pdoc['doc_id']:18s} | emb={emb_score:.3f} judge={judge_score:.3f} "
              f"comb={combined:.3f} | risk={risk_score:.3f} {decision:8s} | "
              f"{technique} | {latency_ms}ms")
        if explanation and use_judge:
            print(f"       judge: {explanation[:100]}")

        results.append({
            "doc_id": pdoc["doc_id"],
            "is_poisoned": True,
            "detected": detected,
            "status": status,
            "embedding_score": round(emb_score, 4),
            "judge_score": round(judge_score, 4),
            "combined_score": round(combined, 4),
            "risk_score": round(risk_score, 4),
            "decision": decision,
            "poison_type": pdoc.get("poison_type", ""),
            "poison_technique": technique,
            "evasion_technique": pdoc.get("evasion_technique", ""),
            "target_query": target_query[:80],
            "judge_explanation": explanation[:200] if explanation else "",
            "latency_ms": latency_ms,
            "judge_available": result.judge_available,
            "test_set": "poisoned",
        })

    # --- TEST SET 2: Clean documents FP test ---
    print(f"\n  [TEST SET 2] Clean-only retrieval (FP test, 10 queries)")

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
        simulated_retrieval = [
            {"doc_id": d["doc_id"], "content": d["content"]}
            for d in clean_docs[:5]
        ]

        t0 = time.time()
        result = pipeline.run(simulated_retrieval, user_query=query, use_judge=use_judge)
        latency_ms = int((time.time() - t0) * 1000)
        total_latency += latency_ms

        false_alarm = any(ds.is_suspicious for ds in result.doc_scores)
        rr = result.risk_result
        risk_score = rr.risk_score if rr else 0.0
        decision = rr.decision if rr else "allow"

        if false_alarm:
            fp += 1
            status = "FP"
        else:
            tn += 1
            status = "TN"

        print(f"  [{status}] risk={risk_score:.3f} {decision:8s} | \"{query[:50]}\" | {latency_ms}ms")

        results.append({
            "doc_id": "clean_set",
            "is_poisoned": False,
            "detected": false_alarm,
            "status": status,
            "embedding_score": 0.0,
            "judge_score": 0.0,
            "combined_score": 0.0,
            "risk_score": round(risk_score, 4),
            "decision": decision,
            "poison_type": "",
            "poison_technique": "",
            "evasion_technique": "",
            "target_query": query[:80],
            "judge_explanation": "",
            "latency_ms": latency_ms,
            "judge_available": result.judge_available,
            "test_set": "clean",
        })

    # --- METRICS ---
    total_tests = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    evasion_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    avg_latency = total_latency // total_tests if total_tests > 0 else 0

    print(f"\n{'='*70}")
    print(f"  ADVANCED RAG HYBRID METRICS — {mode_str.upper()}")
    print(f"{'='*70}")
    print(f"  TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")
    print(f"  Precision:     {precision:.4f}")
    print(f"  Recall:        {recall:.4f}")
    print(f"  F1 Score:      {f1:.4f}")
    print(f"  FPR:           {fpr:.4f}")
    print(f"  Evasion rate:  {evasion_rate:.4f} ({fn}/{tp+fn} evaded)")
    print(f"  Avg latency:   {avg_latency}ms/query")

    # Per-technique breakdown
    print(f"\n  PER-TECHNIQUE BREAKDOWN:")
    print(f"  {'Technique':<25s} {'N':>3s} {'Det':>3s} {'Evd':>3s} {'Evd%':>6s} "
          f"{'AvgEmb':>7s} {'AvgJdg':>7s} {'AvgCmb':>7s}")
    print(f"  {'-'*25} {'-'*3} {'-'*3} {'-'*3} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")
    for tech, s in sorted(technique_results.items()):
        evd_pct = s["evaded"] / s["total"] * 100 if s["total"] > 0 else 0
        avg_e = sum(s["emb_scores"]) / len(s["emb_scores"]) if s["emb_scores"] else 0
        avg_j = sum(s["judge_scores"]) / len(s["judge_scores"]) if s["judge_scores"] else 0
        avg_c = sum(s["combined_scores"]) / len(s["combined_scores"]) if s["combined_scores"] else 0
        print(f"  {tech:<25s} {s['total']:>3d} {s['detected']:>3d} {s['evaded']:>3d} "
              f"{evd_pct:>5.1f}% {avg_e:>7.3f} {avg_j:>7.3f} {avg_c:>7.3f}")

    # Evaded attacks detail
    fn_results = [r for r in results if r["is_poisoned"] and not r["detected"]]
    if fn_results:
        print(f"\n  EVADED ATTACKS ({len(fn_results)}):")
        for r in fn_results:
            print(f"    {r['doc_id']} | emb={r['embedding_score']:.3f} "
                  f"judge={r['judge_score']:.3f} comb={r['combined_score']:.3f} | "
                  f"{r['poison_technique']}")

    print(f"{'='*70}")

    # --- SAVE ---
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)

    suffix = "hybrid" if use_judge else "embonly"
    csv_path = _RUNS_DIR / f"rag_advanced_{suffix}_metrics.csv"
    fieldnames = [
        "doc_id", "is_poisoned", "detected", "status",
        "embedding_score", "judge_score", "combined_score",
        "risk_score", "decision", "poison_type", "poison_technique",
        "evasion_technique", "target_query", "judge_explanation",
        "latency_ms", "judge_available", "test_set",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  [Saved] {csv_path}")

    summary = {
        "dataset": "advanced_poison_samples.json",
        "mode": mode_str,
        "use_judge": use_judge,
        "config": {
            "embedding_weight": lj_cfg.get("embedding_weight", 0.5),
            "judge_weight": lj_cfg.get("judge_weight", 0.5),
            "poison_threshold": rag_cfg.get("poison_threshold", 0.45),
            "judge_abstain_threshold": lj_cfg.get("judge_abstain_threshold", 0.15),
            "embedding_override_multiplier": lj_cfg.get("embedding_override_multiplier", 0.85),
        },
        "total_documents": len(documents),
        "poisoned_count": len(poisoned_docs),
        "clean_count": len(clean_docs),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "evasion_rate": round(evasion_rate, 4),
        "avg_latency_ms": avg_latency,
        "technique_breakdown": {
            tech: {
                "total": s["total"], "detected": s["detected"], "evaded": s["evaded"],
                "evasion_rate": round(s["evaded"] / s["total"], 4) if s["total"] > 0 else 0,
                "avg_embedding": round(sum(s["emb_scores"]) / len(s["emb_scores"]), 4) if s["emb_scores"] else 0,
                "avg_judge": round(sum(s["judge_scores"]) / len(s["judge_scores"]), 4) if s["judge_scores"] else 0,
                "avg_combined": round(sum(s["combined_scores"]) / len(s["combined_scores"]), 4) if s["combined_scores"] else 0,
            }
            for tech, s in technique_results.items()
        },
    }
    summary_path = _RUNS_DIR / f"rag_advanced_{suffix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  [Saved] {summary_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced RAG Hybrid Test")
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="Run embedding-only (no LLM judge) for comparison",
    )
    args = parser.parse_args()
    run_advanced_hybrid_test(use_judge=not args.skip_judge)
