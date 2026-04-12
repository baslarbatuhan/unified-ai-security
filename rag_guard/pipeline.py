"""
rag_guard/pipeline.py
============================
Hybrid RAG Guard detection pipeline.

Combines embedding-based detection with LLM-as-a-Judge:
    retrieved_docs → embedding detector → LLM judge → combined risk → context filter

Combined risk formula:
    combined_risk = 0.4 * embedding_score + 0.6 * llm_judge_score

If LLM judge is unavailable (Ollama down), falls back to embedding-only:
    combined_risk = embedding_score

Usage:
    pipeline = RAGGuardPipeline()
    result   = pipeline.run(documents, user_query="What is ML?")
    result.combined_risk       # per-doc combined scores
    result.risk_result         # RetrievalRiskResult for fusion
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import os
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

try:
    from rag_guard.poison_detector import PoisonDetector, DetectionResult
    from rag_guard.llm_judge import LLMJudge, JudgeResult, JudgeBatchResult
    from rag_guard.retrieval_risk_score import (
        RetrievalRiskScorer,
        RetrievalRiskResult,
        DocScore,
        detection_result_to_doc_scores,
    )
    from rag_guard.context_filter import ContextFilter, SanitizationResult
except ImportError:
    from poison_detector import PoisonDetector, DetectionResult
    from llm_judge import LLMJudge, JudgeResult, JudgeBatchResult
    from retrieval_risk_score import (
        RetrievalRiskScorer,
        RetrievalRiskResult,
        DocScore,
        detection_result_to_doc_scores,
    )
    from context_filter import ContextFilter, SanitizationResult


# ---------------------------------------------------------------------------
# Config — load from secure_balanced.yaml if available
# ---------------------------------------------------------------------------
def _load_weights_from_config() -> tuple:
    """Load embedding/judge weights from configs/secure_balanced.yaml."""
    if yaml is None:
        return 0.4, 0.6
    config_paths = [
        Path(__file__).resolve().parent.parent / "configs" / "secure_balanced.yaml",
        Path(os.getcwd()) / "configs" / "secure_balanced.yaml",
    ]
    for p in config_paths:
        if p.exists():
            try:
                with open(p, "r") as f:
                    cfg = yaml.safe_load(f)
                judge_cfg = cfg.get("modules", {}).get("rag_guard", {}).get("llm_judge", {})
                ew = judge_cfg.get("embedding_weight", 0.4)
                jw = judge_cfg.get("judge_weight", 0.6)
                return float(ew), float(jw)
            except Exception:
                pass
    return 0.4, 0.6

_cfg_ew, _cfg_jw = _load_weights_from_config()
DEFAULT_EMBEDDING_WEIGHT = _cfg_ew
DEFAULT_JUDGE_WEIGHT = _cfg_jw


# ---------------------------------------------------------------------------
# Per-document combined result
# ---------------------------------------------------------------------------
@dataclass
class CombinedDocScore:
    """Combined embedding + LLM judge score for a single document."""
    doc_id: str
    embedding_score: float = 0.0
    judge_score: float = 0.0
    combined_score: float = 0.0
    is_suspicious: bool = False
    judge_explanation: str = ""
    judge_available: bool = True


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------
@dataclass
class RAGPipelineResult:
    """Full result of the RAG guard pipeline."""
    # Per-document scores
    doc_scores: List[CombinedDocScore] = field(default_factory=list)

    # Aggregated risk (fusion-ready)
    risk_result: Optional[RetrievalRiskResult] = None

    # Context filtering
    sanitization: Optional[SanitizationResult] = None
    safe_context: str = ""

    # Metadata
    total_docs: int = 0
    suspicious_count: int = 0
    judge_available: bool = False
    model_used: str = ""
    latency_ms: int = 0
    embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT
    judge_weight: float = DEFAULT_JUDGE_WEIGHT

    def to_module_risk_dict(self) -> Dict[str, Any]:
        """Export risk in ModuleRisk-compatible format."""
        if self.risk_result:
            return self.risk_result.to_module_risk_dict()
        return {
            "module": "rag_guard",
            "risk_score": 0.0,
            "confidence": 0.5,
            "decision": "allow",
            "evidence": ["Pipeline not completed"],
            "latency_ms": self.latency_ms,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class RAGGuardPipeline:
    """
    Hybrid RAG Guard pipeline combining embedding detection with LLM judge.

    Stages:
        1. Embedding detection — PoisonDetector (pattern + semantic)
        2. LLM judge — Ollama-based document analysis (if available)
        3. Score combination — 0.4 * embedding + 0.6 * judge
        4. Risk scoring — RetrievalRiskScorer on combined scores
        5. Context filtering — ContextFilter removes/demotes suspicious docs

    If Ollama is not available, gracefully degrades to embedding-only.
    """

    def __init__(
        self,
        detector: Optional[PoisonDetector] = None,
        judge: Optional[LLMJudge] = None,
        risk_scorer: Optional[RetrievalRiskScorer] = None,
        context_filter: Optional[ContextFilter] = None,
        embedding_weight: float = DEFAULT_EMBEDDING_WEIGHT,
        judge_weight: float = DEFAULT_JUDGE_WEIGHT,
        poison_threshold: float = 0.55,
        removal_threshold: Optional[float] = None,
        low_confidence_threshold: float = 0.35,
        min_safe_docs: int = 2,
    ):
        self.detector = detector or PoisonDetector()
        self.judge = judge or LLMJudge()
        self.risk_scorer = risk_scorer or RetrievalRiskScorer(poison_threshold=poison_threshold)
        rt = removal_threshold if removal_threshold is not None else poison_threshold
        self.context_filter = context_filter or ContextFilter(
            detector=self.detector,
            removal_threshold=rt,
            low_confidence_threshold=low_confidence_threshold,
            min_safe_docs=min_safe_docs,
        )
        self.embedding_weight = embedding_weight
        self.judge_weight = judge_weight
        self.poison_threshold = poison_threshold

    def run(
        self,
        documents: List[Dict[str, Any]],
        user_query: str = "general query",
        use_judge: bool = True,
    ) -> RAGPipelineResult:
        """
        Run the full hybrid pipeline on retrieved documents.

        Args:
            documents:  List of dicts with 'doc_id' and 'content'.
            user_query: User's original query for relevance checking.
            use_judge:  Whether to use LLM judge (set False to skip).

        Returns:
            RAGPipelineResult with combined scores, risk, and filtered context.
        """
        t0 = time.time()

        if not documents:
            return RAGPipelineResult(
                safe_context="No documents retrieved.",
                latency_ms=int((time.time() - t0) * 1000),
            )

        # --- Stage 1: Embedding detection ---
        detection: DetectionResult = self.detector.detect(documents)

        # --- Stage 2: LLM judge (if available) ---
        judge_available = False
        judge_results: Dict[str, JudgeResult] = {}
        model_used = ""

        if use_judge:
            try:
                if self.judge.is_available():
                    judge_batch = self.judge.analyze_batch(documents, user_query=user_query)
                    judge_available = True
                    model_used = judge_batch.model_used
                    for jr in judge_batch.results:
                        judge_results[jr.doc_id] = jr
            except Exception as e:
                print(f"[RAGPipeline] LLM judge unavailable: {e}")

        # --- Stage 3: Combine scores ---
        combined_scores: List[CombinedDocScore] = []
        doc_scores_for_risk: List[DocScore] = []
        suspicious_count = 0

        for i, doc_score in enumerate(detection.document_scores):
            doc_id = doc_score.doc_id
            embedding_score = doc_score.poison_score

            if judge_available and doc_id in judge_results:
                jr = judge_results[doc_id]
                judge_score = jr.judge_score
                combined = (
                    self.embedding_weight * embedding_score
                    + self.judge_weight * judge_score
                )
                explanation = jr.explanation
            else:
                # Fallback: embedding only
                judge_score = 0.0
                combined = embedding_score
                explanation = ""

            combined = round(min(combined, 1.0), 4)
            is_suspicious = combined >= self.poison_threshold

            if is_suspicious:
                suspicious_count += 1

            combined_scores.append(CombinedDocScore(
                doc_id=doc_id,
                embedding_score=round(embedding_score, 4),
                judge_score=round(judge_score, 4),
                combined_score=combined,
                is_suspicious=is_suspicious,
                judge_explanation=explanation,
                judge_available=judge_available,
            ))

            doc_scores_for_risk.append(DocScore(
                doc_id=doc_id,
                poison_score=combined,
                rank=i + 1,
            ))

        # --- Stage 4: Risk scoring ---
        risk_result = self.risk_scorer.score(doc_scores_for_risk)

        # --- Stage 4b: Judge signal amplification ---
        # When the LLM judge flags a document strongly, ensure the overall
        # risk score is not diluted by clean documents in a multi-doc set.
        if judge_available:
            max_judge = max((ds.judge_score for ds in combined_scores), default=0.0)
            if max_judge >= 0.50:
                floor = round(max_judge, 4)
            elif max_judge >= 0.20:
                floor = round(max_judge * 0.80, 4)
            else:
                floor = 0.0
            if floor > 0.0 and risk_result.risk_score < floor:
                risk_result.evidence.append(
                    f"Judge amplification: floor={floor:.4f} (max_judge={max_judge:.4f})"
                )
                risk_result.risk_score = floor
                risk_result.decision = self.risk_scorer.decide(risk_result.risk_score)

        # Add judge info to evidence
        if judge_available:
            risk_result.evidence.append(f"LLM judge used: {model_used}")
            risk_result.evidence.append(
                f"Weights: embedding={self.embedding_weight}, judge={self.judge_weight}"
            )
        else:
            risk_result.evidence.append("LLM judge unavailable — embedding-only scoring")

        # --- Stage 5: Context filtering (same hybrid scores as risk — no second detect())
        combined_for_filter = [ds.combined_score for ds in combined_scores]
        sanitization = self.context_filter.sanitize_with_combined_scores(
            documents, combined_for_filter
        )

        if sanitization.insufficient_clean_docs:
            pol = sanitization.context_policy
            if pol == "block":
                risk_result.risk_score = max(risk_result.risk_score, 0.88)
            elif pol == "sanitize":
                risk_result.risk_score = max(risk_result.risk_score, 0.50)
            risk_result.decision = self.risk_scorer.decide(risk_result.risk_score)
            risk_result.evidence.append(
                f"Context policy: insufficient clean docs "
                f"({sanitization.safe_clean_count}) → {pol}"
            )

        latency_ms = int((time.time() - t0) * 1000)

        return RAGPipelineResult(
            doc_scores=combined_scores,
            risk_result=risk_result,
            sanitization=sanitization,
            safe_context=sanitization.filtered_context,
            total_docs=len(documents),
            suspicious_count=suspicious_count,
            judge_available=judge_available,
            model_used=model_used,
            latency_ms=latency_ms,
            embedding_weight=self.embedding_weight,
            judge_weight=self.judge_weight,
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
    clean = [d for d in documents if not d["is_poisoned"]][:3]
    poisoned = [d for d in documents if d["is_poisoned"]][:2]
    mixed = clean + poisoned

    pipeline = RAGGuardPipeline()

    print(f"{'='*70}")
    print(f"  RAG GUARD HYBRID PIPELINE DEMO")
    print(f"{'='*70}")

    # Run on mixed docs
    result = pipeline.run(mixed, user_query="What is machine learning?")

    print(f"\n  Judge available: {result.judge_available}")
    if result.model_used:
        print(f"  Model: {result.model_used}")
    print(f"  Weights: embedding={result.embedding_weight}, judge={result.judge_weight}")
    print(f"  Latency: {result.latency_ms}ms")

    print(f"\n  Per-document scores:")
    for ds in result.doc_scores:
        status = "SUSPICIOUS" if ds.is_suspicious else "CLEAN"
        judge_info = f"judge={ds.judge_score:.3f}" if ds.judge_available else "judge=N/A"
        print(f"    [{status:10s}] {ds.doc_id}: emb={ds.embedding_score:.3f}, "
              f"{judge_info}, combined={ds.combined_score:.3f}")
        if ds.judge_explanation:
            print(f"      Explanation: {ds.judge_explanation[:70]}")

    print(f"\n  Risk result:")
    print(f"    Score: {result.risk_result.risk_score:.4f}")
    print(f"    Decision: {result.risk_result.decision}")
    print(f"    Flagged: {result.risk_result.flagged_docs}/{result.total_docs}")

    print(f"\n  Context filtering:")
    print(f"    Kept: {result.sanitization.kept_count} | Removed: {result.sanitization.removed_count}")
    print(f"    Context length: {len(result.safe_context)} chars")

    print(f"\n  ModuleRisk: {json.dumps(result.to_module_risk_dict(), indent=4)}")
    print(f"\n{'='*70}")
