"""
rag_guard/build_safe_context.py
================================
High-level safe context builder for RAG pipelines.

Purpose:
    Takes raw retrieved documents, runs them through ContextFilter,
    and produces a ready-to-use safe context string + risk metadata.

    Encapsulates the full flow:
        retrieved_docs → poison detection → filter → rank → safe context

Usage:
    builder = SafeContextBuilder()
    result  = builder.build(retrieved_docs, query="user question")
    llm_context = result.safe_context      # ready for LLM
    risk_info   = result.risk_summary      # for logging/fusion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from rag_guard.context_filter import ContextFilter, SanitizationResult
    from rag_guard.poison_detector import PoisonDetector
except ImportError:
    from context_filter import ContextFilter, SanitizationResult
    from poison_detector import PoisonDetector


@dataclass
class SafeContextResult:
    """Output of SafeContextBuilder.build()."""
    safe_context: str
    doc_count_original: int = 0
    doc_count_kept: int = 0
    doc_count_removed: int = 0
    needs_review: bool = False
    max_poison_score: float = 0.0
    avg_poison_score: float = 0.0
    evidence: List[str] = field(default_factory=list)

    @property
    def risk_summary(self) -> Dict[str, Any]:
        """Summary dict for logging / fusion gateway."""
        return {
            "original": self.doc_count_original,
            "kept": self.doc_count_kept,
            "removed": self.doc_count_removed,
            "needs_review": self.needs_review,
            "max_poison_score": round(self.max_poison_score, 4),
            "avg_poison_score": round(self.avg_poison_score, 4),
        }


class SafeContextBuilder:
    """
    Builds safe LLM context from retrieved documents.

    Wraps ContextFilter and adds:
    - Query-aware context header
    - Poison score statistics
    - min_safe_docs enforcement (delegated to ContextFilter)
    """

    def __init__(
        self,
        detector: Optional[PoisonDetector] = None,
        removal_threshold: float = 0.55,
        low_confidence_threshold: float = 0.35,
        min_safe_docs: int = 2,
    ):
        self.context_filter = ContextFilter(
            detector=detector,
            removal_threshold=removal_threshold,
            low_confidence_threshold=low_confidence_threshold,
            min_safe_docs=min_safe_docs,
        )

    def build(
        self,
        documents: List[Dict[str, Any]],
        query: Optional[str] = None,
    ) -> SafeContextResult:
        """
        Filter documents and build safe context.

        Args:
            documents: List of dicts with 'doc_id' and 'content'.
            query:     Optional user query (added as context header).

        Returns:
            SafeContextResult with safe_context string and metadata.
        """
        if not documents:
            return SafeContextResult(
                safe_context="No documents available.",
                evidence=["Empty document list"],
            )

        # Run filtering
        sanitization: SanitizationResult = self.context_filter.sanitize(documents)

        # Compute poison score stats
        scores = [fd.poison_score for fd in sanitization.filtered_docs]
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Build context with optional query header
        parts = []
        if query:
            parts.append(f"[Query]: {query}\n")
        parts.append(sanitization.filtered_context)
        safe_context = "\n".join(parts)

        return SafeContextResult(
            safe_context=safe_context,
            doc_count_original=sanitization.original_count,
            doc_count_kept=sanitization.kept_count,
            doc_count_removed=sanitization.removed_count,
            needs_review=sanitization.needs_review,
            max_poison_score=max_score,
            avg_poison_score=avg_score,
            evidence=sanitization.evidence,
        )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "datasets" / "poisoned_corpus" / "poison_samples.json"

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    documents = dataset["documents"]
    builder = SafeContextBuilder(min_safe_docs=2)

    print(f"{'='*60}")
    print(f"  SAFE CONTEXT BUILDER DEMO")
    print(f"{'='*60}")

    # Scenario 1: Mixed docs
    clean = [d for d in documents if not d["is_poisoned"]][:3]
    poisoned = [d for d in documents if d["is_poisoned"]][:2]
    mixed = clean + poisoned

    print(f"\n  [Scenario 1] Mixed (3 clean + 2 poisoned):")
    result = builder.build(mixed, query="What is machine learning?")
    print(f"    Original: {result.doc_count_original}")
    print(f"    Kept: {result.doc_count_kept} | Removed: {result.doc_count_removed}")
    print(f"    Needs review: {result.needs_review}")
    print(f"    Max poison: {result.max_poison_score:.3f} | Avg: {result.avg_poison_score:.3f}")
    print(f"    Context length: {len(result.safe_context)} chars")
    print(f"    Risk summary: {result.risk_summary}")

    # Scenario 2: Only 1 clean doc (below min_safe_docs)
    print(f"\n  [Scenario 2] 1 clean + 3 poisoned (below min_safe_docs=2):")
    few_clean = clean[:1] + [d for d in documents if d["is_poisoned"]][:3]
    result2 = builder.build(few_clean, query="Explain neural networks")
    print(f"    Kept: {result2.doc_count_kept} | Removed: {result2.doc_count_removed}")
    print(f"    Needs review: {result2.needs_review}")
    for e in result2.evidence:
        print(f"    {e}")

    # Scenario 3: Empty
    print(f"\n  [Scenario 3] Empty document list:")
    result3 = builder.build([])
    print(f"    Context: \"{result3.safe_context}\"")

    print(f"\n{'='*60}")
