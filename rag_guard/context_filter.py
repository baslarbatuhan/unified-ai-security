"""
rag_guard/context_filter.py
================================
Context sanitization for RAG retrieval results.

Purpose:
    - Instead of blocking entirely when poison is detected, FILTER the results
    - Remove/replace poisoned documents, keep clean ones
    - Provide safe context to LLM rather than empty context

Strategy:
    1. Run poison_detector on retrieved documents
    2. Remove documents flagged as suspicious
    3. If too many removed, flag for manual review
    4. Return filtered context + sanitization report

Usage:
    filter = ContextFilter(detector)
    result = filter.sanitize(retrieved_docs)
    clean_context = result.filtered_context
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

ContextPolicy = Literal["none", "sanitize", "block"]

try:
    from rag_guard.poison_detector import PoisonDetector, DetectionResult
except ImportError:
    from poison_detector import PoisonDetector, DetectionResult


@dataclass
class FilteredDoc:
    """A document after filtering."""
    doc_id: str
    content: str
    is_original: bool     # True if kept as-is, False if sanitized/removed
    poison_score: float
    action: str           # "kept", "removed", "demoted"


@dataclass
class SanitizationResult:
    """Result of context sanitization."""
    original_count: int
    kept_count: int
    removed_count: int
    filtered_docs: List[FilteredDoc] = field(default_factory=list)
    filtered_context: str = ""
    sanitization_ratio: float = 0.0
    needs_review: bool = False
    evidence: List[str] = field(default_factory=list)
    insufficient_clean_docs: bool = False
    context_policy: ContextPolicy = "none"
    safe_clean_count: int = 0


class ContextFilter:
    """
    Filters poisoned documents from RAG retrieval results.

    Instead of blocking the entire request when poison is detected:
    - Removes suspicious documents from the context
    - Keeps clean documents for the LLM
    - If more than half the docs are removed, flags for review
    - Never passes an empty context (falls back to disclaimer)
    """

    def __init__(
        self,
        detector: Optional[PoisonDetector] = None,
        removal_threshold: float = 0.55,
        low_confidence_threshold: float = 0.35,
        review_ratio: float = 0.50,
        min_safe_docs: int = 2,
    ):
        """
        Args:
            detector:                 PoisonDetector instance
            removal_threshold:        Poison score above which a doc is removed
            low_confidence_threshold: Score between this and removal_threshold → demoted (kept but downranked)
            review_ratio:             If more than this ratio is removed, flag for review
            min_safe_docs:            Minimum clean docs required; if fewer, flag for review
        """
        self.detector = detector or PoisonDetector()
        self.removal_threshold = removal_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.review_ratio = review_ratio
        self.min_safe_docs = min_safe_docs

    def sanitize(self, documents: List[Dict]) -> SanitizationResult:
        """
        Filter retrieved documents, removing poisoned content.

        Args:
            documents: List of dicts with 'doc_id' and 'content' keys.

        Returns:
            SanitizationResult with filtered context and report.
        """
        if not documents:
            return SanitizationResult(
                original_count=0, kept_count=0, removed_count=0,
                filtered_context="No documents retrieved.",
                evidence=["Empty retrieval result"],
                insufficient_clean_docs=False,
                context_policy="none",
                safe_clean_count=0,
            )

        # Run poison detection
        detector_input = [{"doc_id": d.get("doc_id", f"doc_{i}"), "content": d.get("content", "")}
                          for i, d in enumerate(documents)]
        detection = self.detector.detect(detector_input)

        # Filter based on scores — three tiers:
        #   score >= removal_threshold       → removed
        #   score >= low_confidence_threshold → demoted (kept but downranked)
        #   score < low_confidence_threshold  → kept as-is
        filtered_docs = []
        kept_docs = []       # clean docs (kept as-is)
        demoted_docs = []    # low confidence docs (kept but after clean docs)
        removed_count = 0
        demoted_count = 0
        evidence = []

        for i, doc_score in enumerate(detection.document_scores):
            doc = documents[i]
            doc_id = doc.get("doc_id", f"doc_{i}")
            content = doc.get("content", "")

            if doc_score.poison_score >= self.removal_threshold:
                # Remove this document
                filtered_docs.append(FilteredDoc(
                    doc_id=doc_id, content="[REMOVED: Suspicious content detected]",
                    is_original=False, poison_score=doc_score.poison_score, action="removed",
                ))
                removed_count += 1
                evidence.append(
                    f"Removed '{doc_id}': poison_score={doc_score.poison_score:.3f} "
                    f"(threshold={self.removal_threshold})"
                )
                if doc_score.pattern_matches:
                    evidence.append(f"  Patterns: {', '.join(doc_score.pattern_matches)}")

            elif doc_score.poison_score >= self.low_confidence_threshold:
                # Demote: keep but place after clean docs
                filtered_docs.append(FilteredDoc(
                    doc_id=doc_id, content=content,
                    is_original=False, poison_score=doc_score.poison_score, action="demoted",
                ))
                demoted_docs.append(content)
                demoted_count += 1
                evidence.append(
                    f"Demoted '{doc_id}': poison_score={doc_score.poison_score:.3f} "
                    f"(low confidence zone {self.low_confidence_threshold}-{self.removal_threshold})"
                )
            else:
                # Keep this document
                filtered_docs.append(FilteredDoc(
                    doc_id=doc_id, content=content,
                    is_original=True, poison_score=doc_score.poison_score, action="kept",
                ))
                kept_docs.append(content)

        # Build filtered context: clean docs first, demoted docs after
        all_safe = kept_docs + demoted_docs
        if all_safe:
            context_parts = [f"[Document {i+1}]: {content}" for i, content in enumerate(all_safe)]
            filtered_context = "\n\n".join(context_parts)
        else:
            filtered_context = "All retrieved documents were flagged as suspicious. Please verify your query or consult the source materials directly."

        # Check if review needed
        original_count = len(documents)
        sanitization_ratio = removed_count / original_count if original_count > 0 else 0
        safe_count = len(kept_docs)
        needs_review = sanitization_ratio >= self.review_ratio

        insufficient_clean = False
        context_policy: ContextPolicy = "none"
        if safe_count < self.min_safe_docs and original_count >= self.min_safe_docs:
            insufficient_clean = True
            needs_review = True
            context_policy = "block" if safe_count == 0 else "sanitize"
            evidence.append(
                f"INSUFFICIENT CLEAN DOCS: {safe_count} fully clean docs < "
                f"min_safe_docs={self.min_safe_docs} → policy={context_policy}."
            )

        if needs_review and sanitization_ratio >= self.review_ratio:
            evidence.append(
                f"HIGH REMOVAL RATE: {removed_count}/{original_count} documents removed "
                f"({sanitization_ratio:.0%}). Manual review recommended."
            )

        if demoted_count > 0:
            evidence.append(
                f"{demoted_count} document(s) demoted (low confidence, placed after clean docs)"
            )

        if removed_count == 0 and demoted_count == 0:
            evidence.append("No documents removed or demoted. All content passed poison detection.")

        return SanitizationResult(
            original_count=original_count,
            kept_count=safe_count + demoted_count,
            removed_count=removed_count,
            filtered_docs=filtered_docs,
            filtered_context=filtered_context,
            sanitization_ratio=round(sanitization_ratio, 4),
            needs_review=needs_review,
            evidence=evidence,
            insufficient_clean_docs=insufficient_clean,
            context_policy=context_policy,
            safe_clean_count=safe_count,
        )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "rag_guard" else Path(__file__).resolve().parent
    dataset_path = project_root / "datasets" / "poisoned_corpus" / "poison_samples.json"

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    documents = dataset["documents"]
    ctx_filter = ContextFilter()

    print(f"{'='*60}")
    print(f"  CONTEXT FILTER DEMO")
    print(f"{'='*60}")

    # Scenario 1: Mixed docs (3 clean + 2 poisoned)
    clean = [d for d in documents if not d["is_poisoned"]][:3]
    poisoned = [d for d in documents if d["is_poisoned"]][:2]
    mixed = clean + poisoned

    print(f"\n  [Scenario 1] Mixed (3 clean + 2 poisoned):")
    result = ctx_filter.sanitize(mixed)
    print(f"    Original: {result.original_count} | Kept: {result.kept_count} | Removed: {result.removed_count}")
    print(f"    Needs review: {result.needs_review}")
    for fd in result.filtered_docs:
        print(f"    {fd.doc_id}: {fd.action} (score={fd.poison_score:.3f})")
    print(f"    Context length: {len(result.filtered_context)} chars")

    # Scenario 2: All clean
    print(f"\n  [Scenario 2] All clean (5 docs):")
    result2 = ctx_filter.sanitize(clean[:5] if len(clean) >= 5 else clean)
    print(f"    Original: {result2.original_count} | Kept: {result2.kept_count} | Removed: {result2.removed_count}")

    # Scenario 3: Mostly poisoned
    print(f"\n  [Scenario 3] Mostly poisoned (1 clean + 4 poisoned):")
    mostly_poison = clean[:1] + [d for d in documents if d["is_poisoned"]][:4]
    result3 = ctx_filter.sanitize(mostly_poison)
    print(f"    Original: {result3.original_count} | Kept: {result3.kept_count} | Removed: {result3.removed_count}")
    print(f"    Needs review: {result3.needs_review}")
    for e in result3.evidence:
        print(f"    {e}")

    print(f"\n{'='*60}")
