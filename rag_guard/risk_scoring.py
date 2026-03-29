"""
rag_guard/risk_scoring.py
==========================
Retrieval Risk Scoring for RAG pipeline.

Purpose:
    - Aggregate poison detection scores into a single risk assessment
    - Produce output compatible with schemas/risk_schema.py (ModuleRisk)
    - Make allow/sanitize/block decisions based on configurable thresholds

Integration:
    poison_detector.py ──► risk_scoring.py ──► fusion_gateway
                                │
                         ModuleRisk output

Scoring Strategy:
    1. Weighted aggregation of per-document poison scores
    2. Poison ratio (how many retrieved docs are suspicious)
    3. Max poison score (worst-case single document)
    4. Combined into final risk_score with confidence

Thresholds (from configs/secure_balanced.yaml):
    allow:    risk_score < 0.30
    sanitize: 0.30 <= risk_score < 0.60
    block:    risk_score >= 0.85
    flag:     0.60 <= risk_score < 0.85
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional, Dict, Any

# Import from sibling module (poison_detector)
try:
    from rag_guard.poison_detector import PoisonDetector, DetectionResult, DocumentScore
except ImportError:
    from poison_detector import PoisonDetector, DetectionResult, DocumentScore


# ---------------------------------------------------------------------------
# Type alias matching schemas/risk_schema.py
# ---------------------------------------------------------------------------
Decision = Literal["allow", "sanitize", "block", "flag"]


# ---------------------------------------------------------------------------
# Configuration (mirrors configs/secure_balanced.yaml)
# ---------------------------------------------------------------------------
@dataclass
class RiskThresholds:
    """Threshold configuration for risk decisions."""
    allow_max: float = 0.30
    sanitize_max: float = 0.60
    block_min: float = 0.85

    def decide(self, score: float) -> Decision:
        if score < self.allow_max:
            return "allow"
        elif score < self.sanitize_max:
            return "sanitize"
        elif score >= self.block_min:
            return "block"
        else:
            return "flag"


# ---------------------------------------------------------------------------
# ModuleRisk-compatible output
# ---------------------------------------------------------------------------
@dataclass
class RAGModuleRisk:
    """
    Risk assessment output for the RAG guard module.
    Compatible with schemas.risk_schema.ModuleRisk.
    """
    module: str = "rag_guard"
    risk_score: float = 0.0
    confidence: float = 0.0
    decision: Decision = "allow"
    evidence: List[str] = field(default_factory=list)
    latency_ms: Optional[int] = None

    # RAG-specific details
    total_retrieved: int = 0
    poisoned_count: int = 0
    poison_ratio: float = 0.0
    max_poison_score: float = 0.0
    flagged_doc_ids: List[str] = field(default_factory=list)

    def to_module_risk_dict(self) -> Dict[str, Any]:
        """Export in ModuleRisk-compatible format for fusion gateway."""
        return {
            "module": self.module,
            "risk_score": round(self.risk_score, 4),
            "confidence": round(self.confidence, 4),
            "decision": self.decision,
            "evidence": self.evidence,
            "latency_ms": self.latency_ms,
        }


# ---------------------------------------------------------------------------
# Risk Scoring Engine
# ---------------------------------------------------------------------------
class RAGRiskScorer:
    """
    Computes retrieval risk score from poison detection results.

    Scoring formula:
        risk_score = w_ratio * poison_ratio
                   + w_max   * max_poison_score
                   + w_avg   * avg_poison_score

    Where:
        - poison_ratio:    fraction of retrieved docs flagged as suspicious
        - max_poison_score: highest individual document poison score
        - avg_poison_score: average poison score across all retrieved docs
    """

    def __init__(
        self,
        thresholds: Optional[RiskThresholds] = None,
        weight_ratio: float = 0.35,
        weight_max: float = 0.40,
        weight_avg: float = 0.25,
    ):
        self.thresholds = thresholds or RiskThresholds()
        self.w_ratio = weight_ratio
        self.w_max = weight_max
        self.w_avg = weight_avg

    def score(self, detection_result: DetectionResult) -> RAGModuleRisk:
        """
        Compute risk score from poison detection results.

        Args:
            detection_result: Output from PoisonDetector.detect()

        Returns:
            RAGModuleRisk with decision and evidence.
        """
        t0 = time.time()

        doc_scores = detection_result.document_scores
        total = detection_result.total_documents
        suspicious = detection_result.suspicious_count

        if total == 0:
            return RAGModuleRisk(
                risk_score=0.0,
                confidence=1.0,
                decision="allow",
                evidence=["No documents retrieved"],
                latency_ms=0,
            )

        # Extract score values
        scores = [ds.poison_score for ds in doc_scores]
        poison_ratio = suspicious / total
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Weighted combination
        risk_score = (
            self.w_ratio * poison_ratio
            + self.w_max * max_score
            + self.w_avg * avg_score
        )
        risk_score = round(min(risk_score, 1.0), 4)

        # Decision
        decision = self.thresholds.decide(risk_score)

        # Confidence calculation
        score_variance = (
            sum((s - avg_score) ** 2 for s in scores) / len(scores)
        ) if len(scores) > 1 else 0.0

        if score_variance < 0.01:
            confidence = 0.90  # scores are consistent -> high confidence
        elif score_variance < 0.05:
            confidence = 0.75
        else:
            confidence = 0.60  # high variance -> lower confidence

        # If pattern matches reinforce semantic signals, boost confidence
        pattern_flagged = sum(1 for ds in doc_scores if ds.pattern_matches)
        semantic_flagged = sum(1 for ds in doc_scores if ds.semantic_similarity > 0.5)
        if pattern_flagged > 0 and semantic_flagged > 0:
            confidence = min(confidence + 0.1, 0.98)

        # Collect evidence
        evidence = []
        flagged_ids = []

        if suspicious > 0:
            evidence.append(f"{suspicious}/{total} retrieved documents flagged as suspicious")
            evidence.append(f"Poison ratio: {poison_ratio:.0%}")

        if max_score > 0.7:
            evidence.append(f"High-risk document detected (score: {max_score:.3f})")

        for ds in doc_scores:
            if ds.is_suspicious:
                flagged_ids.append(ds.doc_id)
                if ds.pattern_matches:
                    evidence.append(
                        f"Doc '{ds.doc_id}': pattern matches [{', '.join(ds.pattern_matches)}]"
                    )
                if ds.semantic_similarity > 0.6:
                    evidence.append(
                        f"Doc '{ds.doc_id}': high semantic similarity to known poison ({ds.semantic_similarity:.3f})"
                    )

        if not evidence:
            evidence.append("No suspicious content detected in retrieved documents")

        latency_ms = int((time.time() - t0) * 1000) + detection_result.latency_ms

        return RAGModuleRisk(
            risk_score=risk_score,
            confidence=round(confidence, 2),
            decision=decision,
            evidence=evidence,
            latency_ms=latency_ms,
            total_retrieved=total,
            poisoned_count=suspicious,
            poison_ratio=round(poison_ratio, 4),
            max_poison_score=round(max_score, 4),
            flagged_doc_ids=flagged_ids,
        )


# ---------------------------------------------------------------------------
# Convenience: full pipeline (detect + score)
# ---------------------------------------------------------------------------
def assess_retrieval_risk(
    documents: List[Dict],
    detector: Optional[PoisonDetector] = None,
    scorer: Optional[RAGRiskScorer] = None,
) -> RAGModuleRisk:
    """
    One-call convenience function: detect poison + compute risk score.

    Args:
        documents: List of dicts with 'doc_id' and 'content' keys.
        detector:  Optional pre-initialized PoisonDetector.
        scorer:    Optional pre-initialized RAGRiskScorer.

    Returns:
        RAGModuleRisk ready for the fusion gateway.
    """
    if detector is None:
        detector = PoisonDetector()
    if scorer is None:
        scorer = RAGRiskScorer()

    detection_result = detector.detect(documents)
    return scorer.score(detection_result)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
def main():
    import json
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "rag_guard" else Path(__file__).resolve().parent
    dataset_path = project_root / "datasets" / "poisoned_corpus" / "poison_samples.json"

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    documents = dataset["documents"]

    # Initialize components
    detector = PoisonDetector()
    scorer = RAGRiskScorer()

    # Simulate: user asks a query, retrieval returns a mix of docs
    # Scenario 1: All clean documents (top-5)
    clean_docs = [{"doc_id": d["doc_id"], "content": d["content"]}
                  for d in documents if not d.get("is_poisoned")][:5]
    risk_clean = assess_retrieval_risk(clean_docs, detector, scorer)

    print(f"\n{'='*60}")
    print(f"  SCENARIO 1: All Clean Documents")
    print(f"{'='*60}")
    print(f"  Risk Score: {risk_clean.risk_score:.4f}")
    print(f"  Decision:   {risk_clean.decision}")
    print(f"  Confidence: {risk_clean.confidence}")
    print(f"  Evidence:   {risk_clean.evidence}")

    # Scenario 2: Mixed (3 clean + 2 poisoned)
    poisoned_docs = [{"doc_id": d["doc_id"], "content": d["content"]}
                     for d in documents if d.get("is_poisoned")][:2]
    mixed_docs = clean_docs[:3] + poisoned_docs
    risk_mixed = assess_retrieval_risk(mixed_docs, detector, scorer)

    print(f"\n{'='*60}")
    print(f"  SCENARIO 2: Mixed (3 clean + 2 poisoned)")
    print(f"{'='*60}")
    print(f"  Risk Score: {risk_mixed.risk_score:.4f}")
    print(f"  Decision:   {risk_mixed.decision}")
    print(f"  Confidence: {risk_mixed.confidence}")
    for e in risk_mixed.evidence:
        print(f"    - {e}")

    # Scenario 3: Mostly poisoned (1 clean + 4 poisoned)
    mostly_poison = clean_docs[:1] + [{"doc_id": d["doc_id"], "content": d["content"]}
                                       for d in documents if d.get("is_poisoned")][:4]
    risk_poison = assess_retrieval_risk(mostly_poison, detector, scorer)

    print(f"\n{'='*60}")
    print(f"  SCENARIO 3: Mostly Poisoned (1 clean + 4 poisoned)")
    print(f"{'='*60}")
    print(f"  Risk Score: {risk_poison.risk_score:.4f}")
    print(f"  Decision:   {risk_poison.decision}")
    print(f"  Confidence: {risk_poison.confidence}")
    for e in risk_poison.evidence:
        print(f"    - {e}")

    # Export ModuleRisk format
    print(f"\n{'='*60}")
    print(f"  MODULERISK OUTPUT (for fusion_gateway)")
    print(f"{'='*60}")
    for label, risk in [("Clean", risk_clean), ("Mixed", risk_mixed), ("Poisoned", risk_poison)]:
        print(f"\n  [{label}] {json.dumps(risk.to_module_risk_dict(), indent=4)}")


if __name__ == "__main__":
    main()
