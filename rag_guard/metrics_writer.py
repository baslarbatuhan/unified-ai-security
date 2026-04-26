"""rag_guard/metrics_writer.py
================================
Append per-call rows to the two RAG guard CSVs:

    runs/rag_final_metrics.csv        — one row per RAG pipeline run
    runs/rag_explainability_log.csv   — one row per suspicious doc / chunk

Lives in its own module so `RAGGuardPipeline.run` stays pure; the fusion
engine, the hybrid test harness, and the chunking sweep runner all call
`record_run(...)` after the fact.

Pairs with `rag_guard.chunk_router.ChunkRouter` — route decisions are the
unit of explainability and get one row each. When router is not used, the
writer still accepts a single "virtual chunk" derived from the doc-level
combined score so the explainability file is never empty on a hit.
"""

from __future__ import annotations

import csv
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional

from rag_guard.pipeline import RAGPipelineResult, CombinedDocScore
from rag_guard.chunk_router import RouteDecision, Route


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUNS_DIR = _PROJECT_ROOT / "runs"

METRICS_PATH = _RUNS_DIR / "rag_final_metrics.csv"
EXPLAIN_PATH = _RUNS_DIR / "rag_explainability_log.csv"

_METRICS_FIELDS = [
    "run_id", "case_id", "target_id",
    "total_docs", "suspicious_count",
    "fused_risk", "decision",
    "embedding_weight", "judge_weight",
    "judge_available", "model_used",
    "latency_ms",
    "route_skip", "route_fast_judge", "route_deep_judge", "route_overrides",
    "top_doc_id", "top_doc_combined",
    "evidence_top",
]
_EXPLAIN_FIELDS = [
    "run_id", "case_id", "target_id",
    "doc_id", "chunk_idx",
    "route", "reason",
    "embedding_score", "judge_score", "used_override",
    "combined_score", "is_suspicious",
    "text_preview",
]

_LOCK = Lock()


def _ensure_writer(path: Path, fields: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    f = path.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fields)
    if new_file:
        w.writeheader()
    return f, w


def _virtual_chunk_from_doc(doc: CombinedDocScore) -> RouteDecision:
    """Fallback when the caller didn't run a ChunkRouter — synthesise one
    decision per doc so explainability rows always exist on suspicious hits."""
    if doc.combined_score > 0.55:
        route = Route.DEEP_JUDGE
        reason = "doc-level combined > 0.55 (router not used)"
    elif doc.combined_score > 0.15:
        route = Route.FAST_JUDGE
        reason = "doc-level combined in ambiguous band (router not used)"
    else:
        route = Route.SKIP
        reason = "doc-level combined below skip threshold (router not used)"
    return RouteDecision(
        idx=0,
        route=route,
        embedding_score=doc.embedding_score,
        reason=reason,
        judge_score=doc.judge_score,
        used_override=False,
        text_preview="",
    )


def record_run(
    result: RAGPipelineResult,
    *,
    run_id: str,
    case_id: str = "",
    target_id: str = "",
    per_doc_routes: Optional[Dict[str, List[RouteDecision]]] = None,
    metrics_path: Optional[Path] = None,
    explain_path: Optional[Path] = None,
) -> None:
    """Append one metrics row + N explainability rows under a single lock.

    Args:
        result:          The RAGPipelineResult produced by pipeline.run().
        run_id:          Stable run identifier (hash of config + version).
        case_id:         Per-call identifier (attack suite case id, etc).
        target_id:       External target id for cross-target comparison.
        per_doc_routes:  Optional {doc_id: [RouteDecision]} from ChunkRouter.
                         When missing, one virtual chunk per suspicious doc
                         is synthesised so explainability is never empty.
    """
    m_path = metrics_path or METRICS_PATH
    e_path = explain_path or EXPLAIN_PATH

    risk_dict = result.to_module_risk_dict()
    decision = risk_dict.get("decision", "allow")
    risk_score = float(risk_dict.get("risk_score", 0.0))
    evidence: List[str] = list(risk_dict.get("evidence", []))

    # Route counters across every doc
    counts = {"skip": 0, "fast_judge": 0, "deep_judge": 0, "overrides": 0}
    if per_doc_routes:
        for decisions in per_doc_routes.values():
            for d in decisions:
                counts[d.route.value] += 1
                if d.used_override:
                    counts["overrides"] += 1

    # Top doc = highest combined score
    top_doc: Optional[CombinedDocScore] = None
    if result.doc_scores:
        top_doc = max(result.doc_scores, key=lambda ds: ds.combined_score)

    metrics_row: Dict[str, Any] = {
        "run_id": run_id,
        "case_id": case_id,
        "target_id": target_id,
        "total_docs": result.total_docs,
        "suspicious_count": result.suspicious_count,
        "fused_risk": round(risk_score, 4),
        "decision": decision,
        "embedding_weight": round(result.embedding_weight, 3),
        "judge_weight": round(result.judge_weight, 3),
        "judge_available": int(bool(result.judge_available)),
        "model_used": result.model_used,
        "latency_ms": int(result.latency_ms),
        "route_skip": counts["skip"],
        "route_fast_judge": counts["fast_judge"],
        "route_deep_judge": counts["deep_judge"],
        "route_overrides": counts["overrides"],
        "top_doc_id": top_doc.doc_id if top_doc else "",
        "top_doc_combined": round(top_doc.combined_score, 4) if top_doc else 0.0,
        "evidence_top": " | ".join(evidence)[:200],
    }

    # Explainability rows: one per chunk when router is provided, else
    # one per suspicious doc (virtual chunk).
    explain_rows: List[Dict[str, Any]] = []
    for doc in result.doc_scores:
        decisions: Iterable[RouteDecision]
        if per_doc_routes and doc.doc_id in per_doc_routes:
            decisions = per_doc_routes[doc.doc_id]
        elif doc.is_suspicious:
            decisions = [_virtual_chunk_from_doc(doc)]
        else:
            # Clean doc and no router data → skip to keep the file lean.
            continue

        for d in decisions:
            explain_rows.append({
                "run_id": run_id,
                "case_id": case_id,
                "target_id": target_id,
                "doc_id": doc.doc_id,
                "chunk_idx": d.idx,
                "route": d.route.value,
                "reason": d.reason[:200],
                "embedding_score": round(d.embedding_score, 4),
                "judge_score": (
                    round(d.judge_score, 4) if d.judge_score is not None else ""
                ),
                "used_override": int(bool(d.used_override)),
                "combined_score": round(doc.combined_score, 4),
                "is_suspicious": int(bool(doc.is_suspicious)),
                "text_preview": (d.text_preview or "")[:200],
            })

    with _LOCK:
        f1, w1 = _ensure_writer(m_path, _METRICS_FIELDS)
        try:
            w1.writerow(metrics_row)
        finally:
            f1.close()
        if explain_rows:
            f2, w2 = _ensure_writer(e_path, _EXPLAIN_FIELDS)
            try:
                w2.writerows(explain_rows)
            finally:
                f2.close()


__all__ = ["record_run", "METRICS_PATH", "EXPLAIN_PATH"]
