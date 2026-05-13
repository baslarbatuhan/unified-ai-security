"""tests/test_thesis_report.py
================================
Hafta 15.B.6 — `scripts/generate_thesis_report.py`.

Tests target the pure-fn pieces (aggregation, per-suite breakdown,
tools_summary, markdown emit). Chart rendering uses matplotlib + Agg
backend and is exercised only as a smoke test (no pixel comparisons —
chart content evolves and exact-pixel diffs are brittle).
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


# Import once; the module pins the Agg backend lazily, so this is safe in CI.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from scripts import generate_thesis_report as gtr  # noqa: E402


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
class TestAggregate:
    def _stub_load_rows(self, monkeypatch, rows_per_run):
        """Patch `_load_per_run_rows` to return a static DataFrame per run."""
        def _fake(entry):
            rid = entry["run_id"]
            return pd.DataFrame(rows_per_run.get(rid, []))
        monkeypatch.setattr(gtr, "_load_per_run_rows", _fake)

    def test_empty_runs_short_circuits(self) -> None:
        out = gtr._aggregate([])
        assert out["n_recent_runs"] == 0
        assert out["n_rows"] == 0

    def test_aggregates_decisions_and_confusion(self, monkeypatch) -> None:
        rows = {
            "r1": [
                {"gateway_decision": "block", "expected_decision": "block",
                 "gateway_miss": 0, "gateway_latency_ms": 100, "suite": "a"},
                {"gateway_decision": "allow", "expected_decision": "block",
                 "gateway_miss": 1, "gateway_latency_ms": 200, "suite": "a"},
                {"gateway_decision": "block", "expected_decision": "allow",
                 "gateway_miss": 0, "gateway_latency_ms": 150, "suite": "b"},
                {"gateway_decision": "allow", "expected_decision": "allow",
                 "gateway_miss": 0, "gateway_latency_ms": 80, "suite": "b"},
            ],
        }
        self._stub_load_rows(monkeypatch, rows)
        agg = gtr._aggregate([{"run_id": "r1"}])
        assert agg["n_rows"] == 4
        # Confusion matrix on block class:
        #   TP=1 (block + block), FN=1 (block + allow),
        #   FP=1 (allow + block), TN=1 (allow + allow)
        assert agg["tp"] == 1
        assert agg["fn"] == 1
        assert agg["fp"] == 1
        assert agg["tn"] == 1
        assert agg["miss_count"] == 1
        # Latency stats
        assert agg["avg_latency_ms"] == pytest.approx(132.5)
        # Suite distribution
        assert agg["suite_distribution"]["a"] == 2
        assert agg["suite_distribution"]["b"] == 2

    def test_module_means_when_present(self, monkeypatch) -> None:
        rows = {
            "r1": [
                {"gateway_decision": "allow", "expected_decision": "allow",
                 "gateway_prompt_score": 0.10, "gateway_rag_score": 0.0,
                 "gateway_agency_score": 0.0, "gateway_latency_ms": 50},
                {"gateway_decision": "block", "expected_decision": "block",
                 "gateway_prompt_score": 0.50, "gateway_rag_score": 0.80,
                 "gateway_agency_score": 0.0, "gateway_latency_ms": 100},
            ],
        }
        self._stub_load_rows(monkeypatch, rows)
        agg = gtr._aggregate([{"run_id": "r1"}])
        means = agg["module_mean_scores"]
        assert means["prompt_guard"] == pytest.approx(0.30)
        assert means["rag_guard"] == pytest.approx(0.40)
        assert means["output_agency"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Per-suite breakdown
# ---------------------------------------------------------------------------
class TestPerSuiteBreakdown:
    def test_groups_by_suite_and_computes_f1(self) -> None:
        df = pd.DataFrame([
            {"suite": "agency_social", "gateway_decision": "block",
             "expected_decision": "block", "gateway_miss": 0},
            {"suite": "agency_social", "gateway_decision": "allow",
             "expected_decision": "block", "gateway_miss": 1},
            {"suite": "rag_poisoning", "gateway_decision": "block",
             "expected_decision": "block", "gateway_miss": 0},
            {"suite": "rag_poisoning", "gateway_decision": "block",
             "expected_decision": "block", "gateway_miss": 0},
        ])
        rows = gtr._per_suite_breakdown({"raw_df": df})
        # Sorted by n desc; agency=2, rag=2 → tied; deterministic group order from pandas.
        suites = {r["suite"]: r for r in rows}
        assert suites["agency_social"]["miss"] == 1
        # rag_poisoning: 2 block/block + 0 FP/FN → perfect
        assert suites["rag_poisoning"]["precision"] == 1.0
        assert suites["rag_poisoning"]["recall"] == 1.0
        assert suites["rag_poisoning"]["f1"] == 1.0

    def test_empty_or_missing_suite_column(self) -> None:
        assert gtr._per_suite_breakdown({"raw_df": pd.DataFrame()}) == []
        df_no_suite = pd.DataFrame([{"gateway_decision": "allow"}])
        assert gtr._per_suite_breakdown({"raw_df": df_no_suite}) == []


# ---------------------------------------------------------------------------
# Tools summary — only counts rows where tool_executed=1
# ---------------------------------------------------------------------------
class TestToolsSummary:
    def test_legacy_rows_dont_inflate_error_count(self) -> None:
        """The bug we just fixed: legacy CSV rows without `tool_error`
        column become `astype(str) == 'nan'` and used to count as errors.
        Now we restrict to rows where the tool actually ran."""
        df = pd.DataFrame([
            # 2 legacy rows — tool didn't run, error column is NaN.
            {"tool_executed": 0, "tool_error": None, "tool_latency_ms": None},
            {"tool_executed": 0, "tool_error": None, "tool_latency_ms": None},
            # 1 real successful tool invocation.
            {"tool_executed": 1, "tool_error": "", "tool_latency_ms": 100},
            # 1 real tool invocation that hit a tool-level error.
            {"tool_executed": 1, "tool_error": "yahoo 404", "tool_latency_ms": 250},
        ])
        out = gtr._tools_summary({"raw_df": df})
        assert out is not None
        assert out["executed"] == 2
        assert out["with_error"] == 1     # not 3 — bug fix
        assert out["avg_tool_latency_ms"] == pytest.approx(175.0)

    def test_returns_none_when_no_tool_data(self) -> None:
        # No tool_executed column
        assert gtr._tools_summary({"raw_df": pd.DataFrame([{"x": 1}])}) is None
        # tool_executed column present but all zero
        df = pd.DataFrame([{"tool_executed": 0}, {"tool_executed": 0}])
        assert gtr._tools_summary({"raw_df": df}) is None


# ---------------------------------------------------------------------------
# Markdown emitter — smoke check that all sections render
# ---------------------------------------------------------------------------
class TestMarkdownEmit:
    def test_emits_all_required_sections(self, tmp_path: Path) -> None:
        df = pd.DataFrame([
            {"suite": "rag_poisoning", "gateway_decision": "block",
             "expected_decision": "block", "gateway_miss": 0,
             "gateway_latency_ms": 100, "gateway_prompt_score": 0.5,
             "gateway_rag_score": 0.8, "gateway_agency_score": 0.0},
        ])
        agg = gtr._aggregate.__wrapped__(  # bypass — we have the df already
            [{"run_id": "r1"}]
        ) if hasattr(gtr._aggregate, "__wrapped__") else None
        # Easier: build the agg dict directly with the fields the emitter uses.
        agg = {
            "n_recent_runs": 1, "n_rows": 1,
            "block_count": 1, "sanitize_count": 0, "allow_count": 0,
            "miss_count": 0, "miss_rate": 0.0, "block_rate": 1.0,
            "tp": 1, "fn": 0, "fp": 0, "tn": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0, "fp_rate": 0.0,
            "avg_latency_ms": 100.0, "p50_latency_ms": 100.0, "p95_latency_ms": 100.0,
            "latency_breach_rate": 0.0,
            "suite_distribution": {"rag_poisoning": 1},
            "module_mean_scores": {"prompt_guard": 0.5, "rag_guard": 0.8,
                                    "output_agency": 0.0},
            "raw_df": df,
        }
        out_md = tmp_path / "report.md"
        gtr._emit_markdown(agg, [{"run_id": "r1", "target_id": "x",
                                  "suite": "rag_poisoning", "n_rows": 1,
                                  "ended_at": "2026-05-13T00:00:00+00:00",
                                  "exit_code": 0}],
                          charts={
                              "attack_distribution": None,
                              "per_module_risk": None,
                              "latency_histogram": None,
                              "decision_breakdown": None,
                              "confusion_block": None,
                          },
                          out_md=out_md)
        text = out_md.read_text(encoding="utf-8")
        for heading in (
            "## Executive summary",
            "## Per-suite breakdown",
            "## Per-module average risk",
            "## Latency distribution",
            "## Confusion matrix",
            "## Recommendations",
            "## Appendix: included runs",
        ):
            assert heading in text, f"missing section: {heading}"
        # Score formula transparency line
        assert "Score formula" in text
        # Run id should appear in the appendix table
        assert "r1" in text

    def test_md_table_helper_handles_empty_rows(self) -> None:
        tbl = gtr._md_table(["a", "b"], [])
        # Header + separator only — no data rows.
        lines = tbl.splitlines()
        assert len(lines) == 2
        assert lines[0] == "|a|b|"
        assert lines[1] == "|---|---|"


# ---------------------------------------------------------------------------
# Chart rendering — smoke (PNGs exist + non-zero size)
# ---------------------------------------------------------------------------
class TestChartSmoke:
    def test_charts_render_to_pngs(self, tmp_path: Path) -> None:
        gtr._set_headless_backend()
        charts_dir = tmp_path / "charts"
        charts_dir.mkdir()
        df = pd.DataFrame([
            {"suite": "x", "gateway_decision": "block", "expected_decision": "block",
             "gateway_latency_ms": 200, "gateway_prompt_score": 0.4,
             "gateway_rag_score": 0.5, "gateway_agency_score": 0.0},
            {"suite": "y", "gateway_decision": "allow", "expected_decision": "allow",
             "gateway_latency_ms": 100, "gateway_prompt_score": 0.1,
             "gateway_rag_score": 0.0, "gateway_agency_score": 0.0},
        ])
        agg = {
            "block_count": 1, "sanitize_count": 0, "allow_count": 1,
            "tp": 1, "fn": 0, "fp": 0, "tn": 1,
            "suite_distribution": {"x": 1, "y": 1},
            "module_mean_scores": {"prompt_guard": 0.25, "rag_guard": 0.25,
                                    "output_agency": 0.0},
            "raw_df": df,
            "avg_latency_ms": 150.0, "p50_latency_ms": 150.0, "p95_latency_ms": 195.0,
            "latency_breach_rate": 0.0,
        }
        outs = [
            gtr._render_attack_distribution(agg, charts_dir),
            gtr._render_per_module_risk(agg, charts_dir),
            gtr._render_latency_histogram(agg, charts_dir),
            gtr._render_decision_breakdown(agg, charts_dir),
            gtr._render_confusion_matrix(agg, charts_dir),
        ]
        for p in outs:
            assert p is not None and p.exists() and p.stat().st_size > 1000

    def test_empty_data_returns_none_not_chart(self, tmp_path: Path) -> None:
        gtr._set_headless_backend()
        charts_dir = tmp_path / "charts"
        charts_dir.mkdir()
        empty_agg = {
            "suite_distribution": {},
            "module_mean_scores": {},
            "raw_df": pd.DataFrame(),
            "block_count": 0, "sanitize_count": 0, "allow_count": 0,
            "tp": 0, "fn": 0, "fp": 0, "tn": 0,
            "p50_latency_ms": 0, "p95_latency_ms": 0,
        }
        for fn in (
            gtr._render_attack_distribution,
            gtr._render_per_module_risk,
            gtr._render_latency_histogram,
            gtr._render_decision_breakdown,
            gtr._render_confusion_matrix,
        ):
            assert fn(empty_agg, charts_dir) is None
