"""
rag_guard/retrieval_risk_score.py
====================================
Improved RAG retrieval risk scoring.

New factors vs old risk_scoring.py:
    OLD: risk = 0.35*poison_ratio + 0.40*max_score + 0.25*avg_score
    NEW: risk = 0.25*top_k_ratio + 0.30*max_score + 0.20*consistency + 0.25*positional

Components:
    1. top_k_poison_ratio  (0.25) — fraction of top-k docs flagged
    2. max_poison_score    (0.30) — worst-case single document
    3. consistency_score   (0.20) — score variance (mixed = suspicious)
    4. positional_score    (0.25) — top-ranked docs weighted more
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

# Bridge import for PoisonDetector integration
try:
    from rag_guard.poison_detector import PoisonDetector, DetectionResult, DocumentScore
except ImportError:
    try:
        from poison_detector import PoisonDetector, DetectionResult, DocumentScore
    except ImportError:
        PoisonDetector = None
        DetectionResult = None
        DocumentScore = None

Decision = Literal["allow", "sanitize", "flag", "block"]

THRESHOLDS = {"allow": 0.30, "sanitize": 0.60, "block": 0.85}


def _decide(score: float, thresholds: Optional[Dict[str, float]] = None) -> Decision:
    th = thresholds or THRESHOLDS
    if score < th["allow"]:
        return "allow"
    elif score < th["sanitize"]:
        return "sanitize"
    elif score >= th["block"]:
        return "block"
    else:
        return "flag"


@dataclass
class DocScore:
    doc_id: str
    poison_score: float
    rank: int  # 1-based


@dataclass
class RetrievalRiskResult:
    module: str = "rag_guard"
    risk_score: float = 0.0
    confidence: float = 0.0
    decision: Decision = "allow"
    evidence: List[str] = field(default_factory=list)
    top_k_poison_ratio: float = 0.0
    max_poison_score: float = 0.0
    consistency_score: float = 0.0
    positional_score: float = 0.0
    total_docs: int = 0
    flagged_docs: int = 0
    flagged_doc_ids: List[str] = field(default_factory=list)

    def to_module_risk_dict(self) -> Dict:
        return {
            "module": self.module,
            "risk_score": round(self.risk_score, 4),
            "confidence": round(self.confidence, 4),
            "decision": self.decision,
            "evidence": self.evidence,
            "latency_ms": None,
        }


class RetrievalRiskScorer:
    """Enhanced risk scoring for RAG retrieval results."""

    def __init__(
        self,
        poison_threshold: float = 0.55,
        weights: Optional[Dict] = None,
        decision_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.poison_threshold = poison_threshold
        self.weights = weights or {
            "top_k_ratio": 0.25, "max_score": 0.30,
            "consistency": 0.20, "positional": 0.25,
        }
        self.decision_thresholds = decision_thresholds or dict(THRESHOLDS)

    def decide(self, score: float) -> Decision:
        """Final decision bands aligned with fusion gateway (configs/policy_thresholds)."""
        return _decide(score, self.decision_thresholds)

    def _top_k_poison_ratio(self, scores: List[DocScore]) -> float:
        if not scores: return 0.0
        return sum(1 for s in scores if s.poison_score >= self.poison_threshold) / len(scores)

    def _max_poison_score(self, scores: List[DocScore]) -> float:
        if not scores: return 0.0
        return max(s.poison_score for s in scores)

    def _consistency_score(self, scores: List[DocScore]) -> float:
        if len(scores) < 2:
            # Single doc: consistency cannot be computed, fall back to
            # the document's own poison score so the signal is not lost.
            return scores[0].poison_score if scores else 0.0
        values = [s.poison_score for s in scores]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return min(math.sqrt(variance) / 0.25, 1.0)

    def _positional_score(self, scores: List[DocScore]) -> float:
        if not scores: return 0.0
        total_w, weighted_sum = 0.0, 0.0
        for s in scores:
            w = max(1.0 - (s.rank - 1) * 0.2, 0.1)
            weighted_sum += s.poison_score * w
            total_w += w
        return weighted_sum / total_w if total_w > 0 else 0.0

    def score(self, doc_scores: List[DocScore]) -> RetrievalRiskResult:
        if not doc_scores:
            return RetrievalRiskResult(risk_score=0.0, confidence=0.5, decision="allow",
                                       evidence=["No documents to score"])

        ratio = self._top_k_poison_ratio(doc_scores)
        max_s = self._max_poison_score(doc_scores)
        consist = self._consistency_score(doc_scores)
        positional = self._positional_score(doc_scores)

        # Single-doc: redistribute consistency weight to max_score
        # (consistency is meaningless with 1 document)
        if len(doc_scores) == 1:
            w = {
                "top_k_ratio": 0.25, "max_score": 0.50,
                "consistency": 0.0, "positional": 0.25,
            }
        else:
            w = self.weights
        risk = round(min(w["top_k_ratio"]*ratio + w["max_score"]*max_s +
                         w["consistency"]*consist + w["positional"]*positional, 1.0), 4)

        flagged = [s for s in doc_scores if s.poison_score >= self.poison_threshold]
        confidence = 0.90 if len(doc_scores) >= 5 else 0.80 if len(doc_scores) >= 3 else 0.70

        evidence = []
        if flagged:
            evidence.append(f"{len(flagged)}/{len(doc_scores)} documents flagged")
            for s in flagged[:3]:
                evidence.append(f"  {s.doc_id} (rank={s.rank}, score={s.poison_score:.3f})")
        else:
            evidence.append("No suspicious documents detected")
        evidence.append(f"Components: ratio={ratio:.3f}, max={max_s:.3f}, consistency={consist:.3f}, positional={positional:.3f}")

        return RetrievalRiskResult(
            risk_score=risk, confidence=confidence, decision=self.decide(risk), evidence=evidence,
            top_k_poison_ratio=round(ratio, 4), max_poison_score=round(max_s, 4),
            consistency_score=round(consist, 4), positional_score=round(positional, 4),
            total_docs=len(doc_scores), flagged_docs=len(flagged),
            flagged_doc_ids=[s.doc_id for s in flagged],
        )


# ---------------------------------------------------------------------------
# Bridge: PoisonDetector → RetrievalRiskScorer
# ---------------------------------------------------------------------------
def detection_result_to_doc_scores(detection: "DetectionResult") -> List[DocScore]:
    """
    Convert PoisonDetector's DetectionResult into List[DocScore]
    that RetrievalRiskScorer.score() expects.

    Args:
        detection: Output from PoisonDetector.detect()

    Returns:
        List of DocScore with rank assigned by retrieval order (1-based).
    """
    return [
        DocScore(
            doc_id=ds.doc_id,
            poison_score=ds.poison_score,
            rank=i + 1,
        )
        for i, ds in enumerate(detection.document_scores)
    ]


def assess_retrieval_risk_v2(
    documents: List[Dict],
    detector: Optional["PoisonDetector"] = None,
    scorer: Optional[RetrievalRiskScorer] = None,
) -> RetrievalRiskResult:
    """
    Convenience function: documents → detect → score in one call.

    Args:
        documents: List of dicts with 'doc_id' and 'content' keys.
        detector:  PoisonDetector instance (creates one if None).
        scorer:    RetrievalRiskScorer instance (creates one if None).

    Returns:
        RetrievalRiskResult ready for fusion gateway via to_module_risk_dict().
    """
    if PoisonDetector is None:
        raise ImportError("PoisonDetector not available. Install: pip install sentence-transformers")

    if detector is None:
        detector = PoisonDetector()
    if scorer is None:
        scorer = RetrievalRiskScorer()

    detection = detector.detect(documents)
    doc_scores = detection_result_to_doc_scores(detection)
    return scorer.score(doc_scores)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    scorer = RetrievalRiskScorer()
    print(f"{'='*60}\n  RETRIEVAL RISK SCORER v2 DEMO\n{'='*60}")

    scenarios = [
        ("All clean", [DocScore(f"doc_{i}", 0.05+i*0.03, i) for i in range(1,6)]),
        ("Top-1 poisoned", [DocScore("poison_1",0.85,1)] + [DocScore(f"doc_{i}",0.10,i) for i in range(2,6)]),
        ("3/5 poisoned", [DocScore("p1",0.80,1),DocScore("d2",0.10,2),DocScore("p3",0.70,3),DocScore("d4",0.15,4),DocScore("p5",0.65,5)]),
        ("All poisoned", [DocScore(f"p_{i}", 0.75+i*0.03, i) for i in range(1,6)]),
    ]
    for name, docs in scenarios:
        r = scorer.score(docs)
        print(f"\n  [{name}] risk={r.risk_score:.3f} | decision={r.decision} | flagged={r.flagged_docs}")
        print(f"    ModuleRisk: {r.to_module_risk_dict()}")
    print(f"\n{'='*60}")
