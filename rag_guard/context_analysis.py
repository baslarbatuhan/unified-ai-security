"""
rag_guard/context_analysis.py
================================
Retrieval context analysis for RAG guard.

Purpose:
    - Analyze retrieved documents for poisoned content
    - Compute: poisoned document ratio, clean retrieval accuracy
    - Measure RAG guard's impact on system performance
    - Report top-k poison ratios (top-1, top-3, top-5)

Integration:
    rag_baseline.py (retrieve) ──► context_analysis.py (analyze) ──► metrics

Dependencies:
    Requires poison_detector.py and rag_baseline.py from the same package.

Usage:
    python rag_guard/context_analysis.py
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    from rag_guard.poison_detector import PoisonDetector, DetectionResult
except ImportError:
    from poison_detector import PoisonDetector, DetectionResult


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FILE_DIR.parent if _FILE_DIR.name == "rag_guard" else _FILE_DIR
_DATASET_PATH = _PROJECT_ROOT / "datasets" / "poisoned_corpus" / "poison_samples.json"
_RUNS_DIR = _PROJECT_ROOT / "runs"


# ---------------------------------------------------------------------------
# Analysis result
# ---------------------------------------------------------------------------
@dataclass
class ContextAnalysisResult:
    """Analysis result for a single retrieval query."""
    query: str
    top_k: int
    total_retrieved: int
    poisoned_detected: int
    clean_count: int
    poison_ratio: float
    top1_poisoned: bool
    top3_poison_ratio: float
    top5_poison_ratio: float
    detection_latency_ms: int
    doc_scores: List[Dict] = field(default_factory=list)


@dataclass
class CorpusAnalysisReport:
    """Aggregated analysis across all queries."""
    total_queries: int
    avg_poison_ratio: float
    avg_top1_asr: float
    avg_top3_poison_ratio: float
    avg_top5_poison_ratio: float
    clean_retrieval_accuracy: float
    total_latency_ms: int
    per_query_results: List[ContextAnalysisResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Context Analyzer
# ---------------------------------------------------------------------------
class ContextAnalyzer:
    """
    Analyzes RAG retrieval context for poison contamination.

    Computes:
    - Poisoned document ratio per query
    - Clean retrieval accuracy (queries with zero poisoned docs)
    - Top-k poison ratios (top-1, top-3, top-5)
    - Per-document poison scores from PoisonDetector
    """

    def __init__(self, detector: Optional[PoisonDetector] = None):
        self.detector = detector or PoisonDetector()

    def analyze_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict],
        top_k: int = 5,
    ) -> ContextAnalysisResult:
        """
        Analyze a single retrieval result for poison contamination.

        Args:
            query:          The retrieval query
            retrieved_docs: List of dicts with 'doc_id', 'content', 'is_poisoned' (ground truth)
            top_k:          Number of docs to analyze

        Returns:
            ContextAnalysisResult with poison metrics.
        """
        docs_to_analyze = retrieved_docs[:top_k]

        # Run poison detector
        detector_input = [{"doc_id": d.get("doc_id", f"doc_{i}"), "content": d.get("content", "")}
                          for i, d in enumerate(docs_to_analyze)]
        detection = self.detector.detect(detector_input)

        # Count poisoned (by detector, not ground truth)
        poisoned_detected = detection.suspicious_count
        clean_count = detection.total_documents - poisoned_detected
        poison_ratio = poisoned_detected / len(docs_to_analyze) if docs_to_analyze else 0.0

        # Top-k poison ratios (by detector)
        top1_poisoned = detection.document_scores[0].is_suspicious if detection.document_scores else False

        top3_docs = detection.document_scores[:3]
        top3_poisoned = sum(1 for d in top3_docs if d.is_suspicious)
        top3_ratio = top3_poisoned / min(3, len(top3_docs)) if top3_docs else 0.0

        top5_docs = detection.document_scores[:5]
        top5_poisoned = sum(1 for d in top5_docs if d.is_suspicious)
        top5_ratio = top5_poisoned / min(5, len(top5_docs)) if top5_docs else 0.0

        # Per-doc scores
        doc_scores = []
        for i, ds in enumerate(detection.document_scores):
            doc_scores.append({
                "doc_id": ds.doc_id,
                "poison_score": ds.poison_score,
                "is_suspicious": ds.is_suspicious,
                "pattern_matches": ds.pattern_matches,
                "semantic_similarity": ds.semantic_similarity,
            })

        return ContextAnalysisResult(
            query=query,
            top_k=top_k,
            total_retrieved=len(docs_to_analyze),
            poisoned_detected=poisoned_detected,
            clean_count=clean_count,
            poison_ratio=round(poison_ratio, 4),
            top1_poisoned=top1_poisoned,
            top3_poison_ratio=round(top3_ratio, 4),
            top5_poison_ratio=round(top5_ratio, 4),
            detection_latency_ms=detection.latency_ms,
            doc_scores=doc_scores,
        )

    def analyze_corpus(
        self,
        queries_and_docs: List[Dict],
        top_k: int = 5,
    ) -> CorpusAnalysisReport:
        """
        Analyze multiple retrieval results across the corpus.

        Args:
            queries_and_docs: List of {"query": str, "retrieved_docs": List[Dict]}
            top_k: Number of docs per query

        Returns:
            CorpusAnalysisReport with aggregated metrics.
        """
        results = []
        total_latency = 0
        poison_ratios = []
        top1_hits = 0
        top3_ratios = []
        top5_ratios = []
        clean_queries = 0

        for item in queries_and_docs:
            result = self.analyze_retrieval(item["query"], item["retrieved_docs"], top_k)
            results.append(result)

            poison_ratios.append(result.poison_ratio)
            top3_ratios.append(result.top3_poison_ratio)
            top5_ratios.append(result.top5_poison_ratio)
            if result.top1_poisoned:
                top1_hits += 1
            if result.poisoned_detected == 0:
                clean_queries += 1
            total_latency += result.detection_latency_ms

        n = len(results) or 1

        return CorpusAnalysisReport(
            total_queries=len(results),
            avg_poison_ratio=round(sum(poison_ratios) / n, 4),
            avg_top1_asr=round(top1_hits / n, 4),
            avg_top3_poison_ratio=round(sum(top3_ratios) / n, 4),
            avg_top5_poison_ratio=round(sum(top5_ratios) / n, 4),
            clean_retrieval_accuracy=round(clean_queries / n, 4),
            total_latency_ms=total_latency,
            per_query_results=results,
        )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
def main():
    if not _DATASET_PATH.exists():
        print(f"Dataset not found: {_DATASET_PATH}")
        return

    with open(_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    documents = dataset["documents"]

    # Initialize analyzer
    analyzer = ContextAnalyzer()

    # Build simulated retrieval scenarios using target queries from poisoned docs
    queries_and_docs = []

    # Scenario A: Target queries (should retrieve poisoned docs)
    for doc in documents:
        if doc.get("is_poisoned") and doc.get("target_query"):
            # Simulate: retrieve this poisoned doc + some clean docs
            clean_docs = [d for d in documents if not d.get("is_poisoned")][:4]
            retrieved = [doc] + clean_docs  # poisoned doc at position 1
            queries_and_docs.append({
                "query": doc["target_query"],
                "retrieved_docs": retrieved,
            })

    # Scenario B: General queries (should only get clean docs)
    general_queries = [
        "How do I deploy a machine learning model?",
        "What is the best branching strategy for Git?",
        "Explain microservices architecture",
    ]
    clean_docs_all = [d for d in documents if not d.get("is_poisoned")]
    for q in general_queries:
        queries_and_docs.append({
            "query": q,
            "retrieved_docs": clean_docs_all[:5],
        })

    # Run analysis
    report = analyzer.analyze_corpus(queries_and_docs)

    # Print results
    print(f"\n{'='*65}")
    print(f"  RAG CONTEXT ANALYSIS REPORT")
    print(f"{'='*65}")
    print(f"  Total queries analyzed:    {report.total_queries}")
    print(f"  Avg poison ratio:          {report.avg_poison_ratio:.3f}")
    print(f"  Top-1 ASR:                 {report.avg_top1_asr:.1%}")
    print(f"  Avg top-3 poison ratio:    {report.avg_top3_poison_ratio:.3f}")
    print(f"  Avg top-5 poison ratio:    {report.avg_top5_poison_ratio:.3f}")
    print(f"  Clean retrieval accuracy:  {report.clean_retrieval_accuracy:.1%}")
    print(f"  Total detection latency:   {report.total_latency_ms}ms")

    print(f"\n  Per-query breakdown:")
    for r in report.per_query_results:
        status = "CONTAMINATED" if r.poisoned_detected > 0 else "CLEAN"
        print(f"    [{status:12s}] ratio={r.poison_ratio:.2f} | "
              f"top1={'P' if r.top1_poisoned else 'C'} | "
              f"top3={r.top3_poison_ratio:.2f} | "
              f"\"{r.query[:50]}\"")

    # Save metrics CSV (metrics_schema format)
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RUNS_DIR / "rag_context_analysis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "module", "test_case", "decision", "risk_score", "latency",
            "query", "poisoned_count", "total_retrieved", "poison_ratio",
            "top1_poisoned", "top3_poison_ratio", "top5_poison_ratio",
        ])
        writer.writeheader()
        for r in report.per_query_results:
            decision = "block" if r.poison_ratio > 0.5 else "flag" if r.poison_ratio > 0.2 else "allow"
            writer.writerow({
                "module": "rag_guard",
                "test_case": r.query[:80],
                "decision": decision,
                "risk_score": r.poison_ratio,
                "latency": r.detection_latency_ms,
                "query": r.query[:80],
                "poisoned_count": r.poisoned_detected,
                "total_retrieved": r.total_retrieved,
                "poison_ratio": r.poison_ratio,
                "top1_poisoned": r.top1_poisoned,
                "top3_poison_ratio": r.top3_poison_ratio,
                "top5_poison_ratio": r.top5_poison_ratio,
            })

    print(f"\n  [Saved] {csv_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
