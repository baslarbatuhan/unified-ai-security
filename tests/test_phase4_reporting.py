"""tests/test_phase4_reporting.py
Unit tests for the `reporting/` package. All inputs are synthetic snapshot
dicts — no telemetry file is read except in the I/O wrapper test, which
points at a tmp path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from reporting.summary_generator import build_summary, render_summary
from reporting.recommendation_engine import (
    derive_recommendations,
    render_recommendations,
)
from reporting.report_generator import render_report, generate_report


# ---------------------------------------------------------------------------
# build_summary / render_summary
# ---------------------------------------------------------------------------
def _base_snapshot(**overrides):
    snap = {
        "total_requests": 100,
        "block_rate": 0.10,
        "sanitize_rate": 0.05,
        "allow_rate": 0.80,
        "attack_bypass_rate": 0.02,
        "avg_latency_ms": 250.0,
        "p95_latency_ms": 800.0,
        "error_rate": 0.0,
        "module_prompt_guard_avg_latency_ms": 50.0,
        "module_rag_guard_avg_latency_ms": 1200.0,
        "module_output_agency_avg_latency_ms": 80.0,
    }
    snap.update(overrides)
    return snap


def test_summary_flag_rate_derived_from_residual():
    s = build_summary(_base_snapshot())
    # block 0.10 + sanitize 0.05 + allow 0.80 → flag = 0.05
    assert s.flag_rate == pytest.approx(0.05, abs=1e-6)


def test_summary_top_alert_picks_highest_severity():
    alerts = [
        {"rule_id": "a", "severity": "info", "message": "x"},
        {"rule_id": "b", "severity": "critical", "message": "y"},
        {"rule_id": "c", "severity": "warn", "message": "z"},
    ]
    s = build_summary(_base_snapshot(), alerts=alerts)
    assert s.top_alert is not None
    assert s.top_alert["rule_id"] == "b"


def test_summary_module_extremes_skip_zero_latency():
    snap = _base_snapshot(module_inactive_avg_latency_ms=0.0)
    s = build_summary(snap)
    # rag_guard slowest, prompt_guard fastest; inactive (0ms) excluded
    assert s.slowest_module == "rag_guard"
    assert s.fastest_module == "prompt_guard"
    assert s.slowest_module_avg_ms == pytest.approx(1200.0)


def test_render_summary_contains_key_headlines():
    s = build_summary(_base_snapshot())
    md = render_summary(s)
    assert "## Executive Summary" in md
    assert "Requests evaluated:** 100" in md
    assert "Bypass proxy" in md


# ---------------------------------------------------------------------------
# derive_recommendations
# ---------------------------------------------------------------------------
def test_recommendations_bypass_fires_critical():
    snap = _base_snapshot(attack_bypass_rate=0.10)
    recs = derive_recommendations(snap)
    ids = [r.rule_id for r in recs]
    assert "tune_fusion_thresholds" in ids
    # First is critical
    assert recs[0].severity == "critical"


def test_recommendations_clean_snapshot_yields_few_or_none():
    # Quiet, healthy system — only "low_block_high_traffic" might fire
    # if total_requests >= 50 and block < 0.01. Use small traffic to skip it.
    snap = _base_snapshot(
        total_requests=10,
        attack_bypass_rate=0.0,
        p95_latency_ms=500.0,
        error_rate=0.0,
        module_rag_guard_avg_latency_ms=400.0,
    )
    recs = derive_recommendations(snap)
    assert all(r.rule_id != "tune_fusion_thresholds" for r in recs)
    assert all(r.rule_id != "investigate_p95_latency" for r in recs)
    assert all(r.rule_id != "high_error_rate" for r in recs)


def test_recommendations_sorted_by_severity():
    snap = _base_snapshot(
        attack_bypass_rate=0.10,        # critical
        p95_latency_ms=3000.0,           # warn
        total_requests=10,               # avoid agency_inactive (needs >=10)
    )
    recs = derive_recommendations(snap)
    severities = [r.severity for r in recs]
    rank = {"critical": 2, "warn": 1, "info": 0}
    assert severities == sorted(severities, key=lambda x: -rank[x])


def test_render_recommendations_empty():
    md = render_recommendations([])
    assert "## Recommendations" in md
    assert "No actionable items" in md


# ---------------------------------------------------------------------------
# render_report
# ---------------------------------------------------------------------------
def test_render_report_has_all_sections():
    snap = _base_snapshot(attack_bypass_rate=0.08)
    alerts = [{"rule_id": "bypass", "severity": "critical", "message": "high"}]
    breakers = [
        {"name": "llm_judge", "state": "closed",
         "consecutive_failures": 0, "total_failures": 1, "total_short_circuits": 0}
    ]
    md = render_report(snap, alerts, breakers, event_count=100)
    for header in (
        "# Chatbot Security Report",
        "## Executive Summary",
        "## Per-module performance",
        "## Active alerts",
        "## Circuit breakers",
        "## Recommendations",
        "## Methodology",
    ):
        assert header in md, f"missing section: {header}"


def test_render_report_handles_empty_inputs():
    md = render_report({}, [], [], event_count=0)
    assert "_None._" in md           # no alerts
    assert "_No registered breakers._" in md
    assert "_No module results in window._" in md


# ---------------------------------------------------------------------------
# generate_report (I/O wrapper, with injected events)
# ---------------------------------------------------------------------------
def test_generate_report_writes_file_with_injected_events(tmp_path: Path):
    # Synthetic events shaped like the telemetry stream's fusion records.
    events = [
        {
            "event_type": "fusion_decision",
            "timestamp": "2026-04-25T10:00:00Z",
            "decision": "allow",
            "fused_score": 0.10,
            "module_max": 0.20,
            "latency_ms": 200,
        },
        {
            "event_type": "fusion_decision",
            "timestamp": "2026-04-25T10:00:01Z",
            "decision": "block",
            "fused_score": 0.80,
            "module_max": 0.85,
            "latency_ms": 300,
        },
    ]
    out = tmp_path / "subdir" / "report.md"
    result = generate_report(out, events=events)
    assert result == out
    body = out.read_text(encoding="utf-8")
    assert "# Chatbot Security Report" in body
    assert "## Executive Summary" in body


def test_generate_report_no_events_no_telemetry_file(tmp_path: Path, monkeypatch):
    # Force read_events to raise FileNotFoundError so the wrapper falls back to [].
    import schemas.telemetry_schema as ts_mod

    def boom(*a, **kw):
        raise FileNotFoundError("no telemetry")

    monkeypatch.setattr(ts_mod, "read_events", boom)
    out = tmp_path / "report.md"
    result = generate_report(out)
    assert result.exists()
    body = result.read_text(encoding="utf-8")
    assert "# Chatbot Security Report" in body
