"""tests/test_rag_metrics_routing_cost.py
==========================================
Hafta 12.2 — chunk routing performance/cost telemetry.

Covers:
  * `ChunkRouter.cost_breakdown()` math: total_chunks_evaluated,
    total_llm_judge_calls, routing_savings_pct, phase_ms aggregates.
  * `metrics_writer.record_run` writes the new aggregate columns into
    `rag_final_metrics.csv` and the per-chunk timing columns into
    `rag_explainability_log.csv`.
  * Schema drift: old headers in either file are rotated to
    `.stale-<utc>` and the new write starts with the v12.2 schema.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import pytest

from rag_guard.chunk_router import ChunkRouter, Route, RouteDecision
from rag_guard.metrics_writer import (
    _METRICS_FIELDS,
    _EXPLAIN_FIELDS,
    record_run,
)
from rag_guard.pipeline import RAGPipelineResult, CombinedDocScore


# ---------------------------------------------------------------------------
# 1) ChunkRouter.cost_breakdown
# ---------------------------------------------------------------------------
class TestCostBreakdownMath:
    def test_empty_list_zero_cost(self) -> None:
        d = ChunkRouter.cost_breakdown([])
        assert d["total_chunks_evaluated"] == 0
        assert d["total_llm_judge_calls"] == 0
        assert d["routing_savings_pct"] == 0.0
        assert d["embedding_phase_ms"] == 0
        assert d["judge_phase_ms"] == 0

    def test_mixed_routes_counts_and_savings_pct(self) -> None:
        # 5 chunks: 2 SKIP, 2 FAST, 1 DEEP → 3 judge calls, 40% savings.
        decs = [
            RouteDecision(idx=0, route=Route.SKIP, embedding_score=0.05,
                          reason="r", embedding_time_ms=10),
            RouteDecision(idx=1, route=Route.SKIP, embedding_score=0.10,
                          reason="r", embedding_time_ms=10),
            RouteDecision(idx=2, route=Route.FAST_JUDGE, embedding_score=0.30,
                          reason="r", embedding_time_ms=15, judge_time_ms=500),
            RouteDecision(idx=3, route=Route.FAST_JUDGE, embedding_score=0.35,
                          reason="r", embedding_time_ms=15, judge_time_ms=520),
            RouteDecision(idx=4, route=Route.DEEP_JUDGE, embedding_score=0.80,
                          reason="r", embedding_time_ms=20, judge_time_ms=1500),
        ]
        d = ChunkRouter.cost_breakdown(decs)
        assert d["total_chunks_evaluated"] == 5
        assert d["total_llm_judge_calls"] == 3
        assert d["routing_savings_pct"] == pytest.approx(40.0)
        assert d["embedding_phase_ms"] == 70
        assert d["judge_phase_ms"] == 2520

    def test_all_skip_100_pct_savings(self) -> None:
        decs = [
            RouteDecision(idx=i, route=Route.SKIP, embedding_score=0.05,
                          reason="r", embedding_time_ms=5)
            for i in range(4)
        ]
        d = ChunkRouter.cost_breakdown(decs)
        assert d["total_llm_judge_calls"] == 0
        assert d["routing_savings_pct"] == pytest.approx(100.0)

    def test_all_judge_zero_savings(self) -> None:
        decs = [
            RouteDecision(idx=i, route=Route.DEEP_JUDGE, embedding_score=0.8,
                          reason="r", judge_time_ms=1000)
            for i in range(3)
        ]
        d = ChunkRouter.cost_breakdown(decs)
        assert d["routing_savings_pct"] == 0.0
        assert d["total_llm_judge_calls"] == 3


# ---------------------------------------------------------------------------
# 2) metrics_writer.record_run — new columns end up in the CSV
# ---------------------------------------------------------------------------
def _stub_result() -> RAGPipelineResult:
    """Minimum viable RAGPipelineResult to feed record_run."""
    doc = CombinedDocScore(
        doc_id="d1",
        embedding_score=0.45,
        judge_score=0.60,
        combined_score=0.55,
        is_suspicious=True,
        judge_explanation="",
        judge_available=True,
        chunk_scores=[],
    )
    return RAGPipelineResult(
        total_docs=1,
        suspicious_count=1,
        doc_scores=[doc],
        embedding_weight=0.3,
        judge_weight=0.7,
        judge_available=True,
        model_used="qwen2.5:7b",
        latency_ms=100,
    )


class TestRecordRunNewColumns:
    def test_metrics_csv_has_v12_2_columns(self, tmp_path: Path) -> None:
        m = tmp_path / "rag_final_metrics.csv"
        e = tmp_path / "rag_explainability_log.csv"
        per_doc = {"d1": [
            RouteDecision(idx=0, route=Route.SKIP, embedding_score=0.05,
                          reason="below skip", embedding_time_ms=10),
            RouteDecision(idx=1, route=Route.DEEP_JUDGE, embedding_score=0.8,
                          reason="above deep", embedding_time_ms=15,
                          judge_time_ms=1234, judge_score=0.75,
                          used_override=False),
        ]}
        record_run(
            _stub_result(), run_id="test-run", case_id="c1",
            target_id="mock", per_doc_routes=per_doc,
            metrics_path=m, explain_path=e,
        )

        with m.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert reader.fieldnames is not None
        # All v12.2 columns present.
        for col in ("total_chunks_evaluated", "total_llm_judge_calls",
                    "routing_savings_pct", "embedding_phase_ms", "judge_phase_ms"):
            assert col in reader.fieldnames, f"missing {col}"
        assert len(rows) == 1
        r = rows[0]
        # 2 chunks, 1 SKIP, 1 DEEP → 1 judge call, 50% savings.
        assert int(r["total_chunks_evaluated"]) == 2
        assert int(r["total_llm_judge_calls"]) == 1
        assert float(r["routing_savings_pct"]) == 50.0
        assert int(r["embedding_phase_ms"]) == 25
        assert int(r["judge_phase_ms"]) == 1234

    def test_explain_csv_has_per_chunk_timing(self, tmp_path: Path) -> None:
        m = tmp_path / "rag_final_metrics.csv"
        e = tmp_path / "rag_explainability_log.csv"
        per_doc = {"d1": [
            RouteDecision(idx=0, route=Route.FAST_JUDGE, embedding_score=0.30,
                          reason="ambiguous", embedding_time_ms=12,
                          judge_time_ms=600, judge_score=0.4),
        ]}
        record_run(
            _stub_result(), run_id="test-run", case_id="c1",
            target_id="mock", per_doc_routes=per_doc,
            metrics_path=m, explain_path=e,
        )
        with e.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert "embedding_time_ms" in reader.fieldnames
        assert "judge_time_ms" in reader.fieldnames
        assert int(rows[0]["embedding_time_ms"]) == 12
        assert int(rows[0]["judge_time_ms"]) == 600


# ---------------------------------------------------------------------------
# 3) Schema drift rotation — old header rotates aside
# ---------------------------------------------------------------------------
class TestSchemaDriftRotation:
    def test_old_metrics_header_rotates_to_stale(self, tmp_path: Path) -> None:
        m = tmp_path / "rag_final_metrics.csv"
        e = tmp_path / "rag_explainability_log.csv"

        # Write an old-style metrics file with fewer columns.
        legacy_cols = [c for c in _METRICS_FIELDS
                       if c not in {"total_chunks_evaluated",
                                    "total_llm_judge_calls",
                                    "routing_savings_pct",
                                    "embedding_phase_ms",
                                    "judge_phase_ms"}]
        with m.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=legacy_cols)
            w.writeheader()
            w.writerow({k: "" for k in legacy_cols})

        record_run(
            _stub_result(), run_id="r", case_id="c", target_id="t",
            metrics_path=m, explain_path=e,
        )

        # New file has the current header.
        with m.open(encoding="utf-8") as f:
            new_header = f.readline().strip()
        assert new_header == ",".join(_METRICS_FIELDS)

        # Old data is preserved under `.stale-*`.
        stale = list(tmp_path.glob("rag_final_metrics.csv.stale-*"))
        assert len(stale) == 1

    def test_old_explain_header_rotates_to_stale(self, tmp_path: Path) -> None:
        m = tmp_path / "rag_final_metrics.csv"
        e = tmp_path / "rag_explainability_log.csv"

        legacy_cols = [c for c in _EXPLAIN_FIELDS
                       if c not in {"embedding_time_ms", "judge_time_ms"}]
        with e.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=legacy_cols)
            w.writeheader()
            w.writerow({k: "" for k in legacy_cols})

        per_doc = {"d1": [
            RouteDecision(idx=0, route=Route.FAST_JUDGE, embedding_score=0.3,
                          reason="r", judge_score=0.5),
        ]}
        record_run(
            _stub_result(), run_id="r", case_id="c", target_id="t",
            per_doc_routes=per_doc,
            metrics_path=m, explain_path=e,
        )

        with e.open(encoding="utf-8") as f:
            new_header = f.readline().strip()
        assert new_header == ",".join(_EXPLAIN_FIELDS)
        stale = list(tmp_path.glob("rag_explainability_log.csv.stale-*"))
        assert len(stale) == 1
