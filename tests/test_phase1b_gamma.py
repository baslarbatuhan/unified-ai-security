"""tests/test_phase1b_gamma.py
================================
Phase 1B-γ unit tests — chunk router + RAG final metrics writer.
No LLM, no network.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# ChunkRouter
# ---------------------------------------------------------------------------
def test_router_classifies_three_bands():
    from rag_guard.chunk_router import ChunkRouter, RouterConfig, Route

    r = ChunkRouter(RouterConfig(skip_below=0.15, deep_above=0.55))
    assert r.decide(0, 0.05).route == Route.SKIP
    assert r.decide(1, 0.30).route == Route.FAST_JUDGE
    assert r.decide(2, 0.70).route == Route.DEEP_JUDGE


def test_router_clamps_scores_into_unit_interval():
    from rag_guard.chunk_router import ChunkRouter, Route

    r = ChunkRouter()
    # Values outside [0,1] must not crash or flip bands.
    assert r.decide(0, -0.5).route == Route.SKIP
    assert r.decide(1, 1.5).route == Route.DEEP_JUDGE
    d = r.decide(2, 0.8)
    assert 0.0 <= d.embedding_score <= 1.0


def test_router_config_invariant():
    from rag_guard.chunk_router import RouterConfig

    with pytest.raises(ValueError):
        RouterConfig(skip_below=0.6, deep_above=0.4)


def test_router_decide_batch_reuses_previews():
    from rag_guard.chunk_router import ChunkRouter

    r = ChunkRouter()
    decisions = r.decide_batch([0.05, 0.4, 0.9], chunks=["lorem", "ipsum", "dolor"])
    assert [d.text_preview for d in decisions] == ["lorem", "ipsum", "dolor"]


def test_router_finalize_applies_deep_override():
    from rag_guard.chunk_router import ChunkRouter, Route

    r = ChunkRouter()
    # Deep route with judge "abstain" — embedding override should dominate.
    d = r.decide(0, 0.80)
    assert d.route == Route.DEEP_JUDGE
    effective = r.finalize(d, judge_score=0.30)
    assert effective > 0.30  # override engaged
    assert d.used_override is True
    assert d.judge_score == pytest.approx(0.30)


def test_router_finalize_skip_forces_zero():
    from rag_guard.chunk_router import ChunkRouter

    r = ChunkRouter()
    d = r.decide(0, 0.05)
    effective = r.finalize(d, judge_score=0.9)  # caller shouldn't call judge
    assert effective == 0.0
    assert d.judge_score == 0.0
    assert d.used_override is False


def test_router_summary_counts():
    from rag_guard.chunk_router import ChunkRouter, Route

    r = ChunkRouter()
    decisions = [r.decide(i, s) for i, s in enumerate([0.05, 0.3, 0.6, 0.9])]
    for d in decisions:
        if d.route == Route.DEEP_JUDGE:
            r.finalize(d, 0.2)  # trigger override
        else:
            r.finalize(d, 0.5)
    summary = ChunkRouter.summarize(decisions)
    assert summary["skip"] == 1
    assert summary["fast_judge"] == 1
    assert summary["deep_judge"] == 2
    assert summary["overrides"] >= 1


# ---------------------------------------------------------------------------
# RAG metrics writer
# ---------------------------------------------------------------------------
def _make_result(suspicious: bool, decision: str = "block", risk: float = 0.82):
    """Build a minimal RAGPipelineResult without touching the real pipeline."""
    from rag_guard.pipeline import RAGPipelineResult, CombinedDocScore
    from rag_guard.retrieval_risk_score import RetrievalRiskResult

    docs = [
        CombinedDocScore(
            doc_id="doc_clean",
            embedding_score=0.08,
            judge_score=0.05,
            combined_score=0.065,
            is_suspicious=False,
        ),
        CombinedDocScore(
            doc_id="doc_bad",
            embedding_score=0.72 if suspicious else 0.12,
            judge_score=0.88 if suspicious else 0.05,
            combined_score=risk if suspicious else 0.09,
            is_suspicious=suspicious,
        ),
    ]
    # Construct a minimal risk_result that supports to_module_risk_dict().
    class _Stub:
        def to_module_risk_dict(self):
            return {
                "module": "rag_guard",
                "risk_score": risk if suspicious else 0.05,
                "confidence": 0.9,
                "decision": decision if suspicious else "allow",
                "evidence": ["Poisoned doc detected" if suspicious else "Clean corpus"],
                "latency_ms": 123,
            }
    return RAGPipelineResult(
        doc_scores=docs,
        risk_result=_Stub(),
        total_docs=len(docs),
        suspicious_count=1 if suspicious else 0,
        judge_available=True,
        model_used="qwen2.5:7b",
        latency_ms=123,
        embedding_weight=0.5,
        judge_weight=0.5,
    )


def test_record_run_writes_both_csvs(tmp_path):
    from rag_guard.metrics_writer import (
        record_run, _METRICS_FIELDS, _EXPLAIN_FIELDS,
    )

    m = tmp_path / "rag_final_metrics.csv"
    e = tmp_path / "rag_explainability_log.csv"

    result = _make_result(suspicious=True)
    record_run(
        result,
        run_id="test_run_001",
        case_id="adv_poison_001",
        target_id="mock_target",
        metrics_path=m,
        explain_path=e,
    )

    assert m.exists() and e.exists()
    with m.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert set(rows[0].keys()) == set(_METRICS_FIELDS)
    row = rows[0]
    assert row["run_id"] == "test_run_001"
    assert row["case_id"] == "adv_poison_001"
    assert row["decision"] == "block"
    assert int(row["suspicious_count"]) == 1
    assert row["top_doc_id"] == "doc_bad"
    assert float(row["top_doc_combined"]) > 0.5

    with e.open("r", encoding="utf-8") as f:
        ex_rows = list(csv.DictReader(f))
    # Virtual chunk synthesised for the single suspicious doc.
    assert len(ex_rows) == 1
    assert set(ex_rows[0].keys()) == set(_EXPLAIN_FIELDS)
    assert ex_rows[0]["doc_id"] == "doc_bad"
    assert ex_rows[0]["route"] == "deep_judge"
    assert int(ex_rows[0]["is_suspicious"]) == 1


def test_record_run_with_router_decisions_one_row_per_chunk(tmp_path):
    from rag_guard.metrics_writer import record_run
    from rag_guard.chunk_router import ChunkRouter

    result = _make_result(suspicious=True)
    router = ChunkRouter()

    # 3 chunks for the bad doc, 1 for the clean doc.
    bad_decisions = router.decide_batch(
        [0.10, 0.45, 0.80], chunks=["clean span", "borderline", "poisoned"]
    )
    for d in bad_decisions:
        router.finalize(d, judge_score=0.5)

    clean_decisions = router.decide_batch([0.05], chunks=["hello"])
    router.finalize(clean_decisions[0], judge_score=0.0)

    m = tmp_path / "m.csv"
    e = tmp_path / "e.csv"
    record_run(
        result, run_id="r1",
        per_doc_routes={"doc_bad": bad_decisions, "doc_clean": clean_decisions},
        metrics_path=m, explain_path=e,
    )

    with m.open("r", encoding="utf-8") as f:
        metrics_row = list(csv.DictReader(f))[0]
    # 4 total chunk decisions routed: 2 skip + 1 fast + 1 deep
    assert int(metrics_row["route_skip"]) == 2
    assert int(metrics_row["route_fast_judge"]) == 1
    assert int(metrics_row["route_deep_judge"]) == 1

    with e.open("r", encoding="utf-8") as f:
        ex_rows = list(csv.DictReader(f))
    # All 4 chunks recorded (even clean doc's, because router gave decisions).
    assert len(ex_rows) == 4
    routes_by_doc = {}
    for r in ex_rows:
        routes_by_doc.setdefault(r["doc_id"], []).append(r["route"])
    assert sorted(routes_by_doc["doc_bad"]) == ["deep_judge", "fast_judge", "skip"]
    assert routes_by_doc["doc_clean"] == ["skip"]


def test_record_run_benign_no_explain_rows(tmp_path):
    """No router, no suspicious docs → explain file should remain absent."""
    from rag_guard.metrics_writer import record_run

    result = _make_result(suspicious=False, decision="allow", risk=0.05)
    m = tmp_path / "m.csv"
    e = tmp_path / "e.csv"
    record_run(result, run_id="r1", metrics_path=m, explain_path=e)
    assert m.exists()
    # No suspicious docs, no router → explainability file never opened.
    assert not e.exists()


def test_record_run_appends_on_second_call(tmp_path):
    from rag_guard.metrics_writer import record_run

    m = tmp_path / "m.csv"
    e = tmp_path / "e.csv"
    record_run(_make_result(True), run_id="r", case_id="c1",
               metrics_path=m, explain_path=e)
    record_run(_make_result(True), run_id="r", case_id="c2",
               metrics_path=m, explain_path=e)
    with m.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["case_id"] for r in rows] == ["c1", "c2"]
