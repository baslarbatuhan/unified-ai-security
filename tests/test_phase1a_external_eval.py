"""tests/test_phase1a_external_eval.py
========================================
Phase 1A unit tests — no LLM, no network (gateway is disabled via flag).

Covers:
  * configs/timeout_loader.py        — profile lookup + budget accessor
  * external_eval/attack_suites.py  — loader shapes, tool filter
  * external_eval/run_external_eval.py — end-to-end against mock adapter
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Timeout config loader
# ---------------------------------------------------------------------------
def test_timeout_profiles_exist():
    from configs.timeout_loader import load_timeout_profile, module_budget_ms

    for profile_name in ("standard", "fast", "generous"):
        profile = load_timeout_profile(profile_name)
        # Every profile must define budgets for the four core modules.
        for mod in ("prompt_guard", "rag_guard", "output_agency", "fusion"):
            assert module_budget_ms(profile, mod) > 0, (
                f"profile {profile_name!r} missing budget for {mod!r}"
            )
        assert profile["total_request_ms"] > 0
        assert profile["adapter"]["api_ms"] > 0


def test_timeout_profile_unknown_raises():
    from configs.timeout_loader import load_timeout_profile

    with pytest.raises(KeyError):
        load_timeout_profile("does_not_exist")


def test_service_limits_loads():
    from configs.timeout_loader import load_service_limits

    limits = load_service_limits()
    assert "rate_limit" in limits
    assert limits["rate_limit"]["default"]["requests_per_minute"] > 0
    assert limits["payload"]["max_prompt_bytes"] > 0
    assert limits["circuit_breaker"]["failure_threshold"] >= 1


# ---------------------------------------------------------------------------
# Attack suite loaders
# ---------------------------------------------------------------------------
def test_load_prompt_injection_suite_nonempty():
    from external_eval.attack_suites import load_prompt_injection

    cases = load_prompt_injection(limit=5)
    assert 0 < len(cases) <= 5
    c = cases[0]
    assert c.suite == "prompt_injection"
    assert c.prompt
    assert c.expected == "block"
    assert c.requires_tools is False


def test_load_rag_poisoning_suite_wraps_context():
    from external_eval.attack_suites import load_rag_poisoning

    cases = load_rag_poisoning(limit=3)
    assert len(cases) == 3
    for c in cases:
        assert c.suite == "rag_poisoning"
        # The prompt must include the doc body as retrieval context.
        assert "supporting document" in c.prompt.lower()
        assert "user question" in c.prompt.lower()
        assert c.requires_tools is False


def test_load_agency_social_marks_requires_tools():
    from external_eval.attack_suites import load_agency_social

    cases = load_agency_social(limit=3)
    assert len(cases) >= 1
    assert all(c.requires_tools for c in cases)
    assert all("tool" in c.metadata for c in cases)


def test_filter_for_target_drops_tool_cases():
    from external_eval.attack_suites import load_suite, filter_for_target

    # limit applies after concatenation; use None so agency_social cases
    # are definitely included.
    all_cases = load_suite("all")
    has_tool_cases = any(c.requires_tools for c in all_cases)
    assert has_tool_cases, "fixture sanity: expected some agency cases to require tools"

    no_tools = filter_for_target(all_cases, target_has_tools=False)
    assert all(not c.requires_tools for c in no_tools)
    with_tools = filter_for_target(all_cases, target_has_tools=True)
    assert len(with_tools) == len(all_cases)


def test_load_suite_unknown_raises():
    from external_eval.attack_suites import load_suite

    with pytest.raises(KeyError):
        load_suite("nonsense_suite")


# ---------------------------------------------------------------------------
# Runner — end-to-end against mock target, gateway disabled.
# ---------------------------------------------------------------------------
def test_run_external_eval_against_mock(tmp_path, monkeypatch):
    """Smoke test: runner produces a valid CSV on the mock target."""
    from schemas import telemetry_schema as ts
    from external_eval import run_external_eval as runner

    # Redirect telemetry so tests don't pollute the real log.
    monkeypatch.setattr(ts, "TELEMETRY_DIR", tmp_path)
    monkeypatch.setattr(ts, "TELEMETRY_FILE", tmp_path / "telemetry.jsonl")

    out_csv = tmp_path / "external_eval_results.csv"
    rc = runner.run(
        [
            "--target", "mock_echo",
            "--suite", "prompt_injection",
            "--max-attacks", "3",
            "--no-gateway-analyze",
            "--output-csv", str(out_csv),
            "--run-id", "test_run_001",
        ]
    )
    assert rc == 0
    assert out_csv.exists()

    with out_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert all(r["target_id"] == "mock_echo" for r in rows)
    assert all(r["run_id"] == "test_run_001" for r in rows)
    assert all(int(r["adapter_ok"]) == 1 for r in rows)
    # Mock always echoes — response_chars > 0.
    assert all(int(r["response_chars"]) > 0 for r in rows)

    # Telemetry: one RequestEvent per case, no FusionDecisionEvent (gateway off).
    events = ts.read_events(run_id="test_run_001")
    assert sum(1 for e in events if e["kind"] == "request") == 3
    assert sum(1 for e in events if e["kind"] == "fusion_decision") == 0


def test_run_external_eval_skips_tool_cases_for_no_tools_target(tmp_path, monkeypatch):
    """agency_social requires tools; without --target-has-tools the runner
    must drop every case and exit non-zero with a clear message."""
    from schemas import telemetry_schema as ts
    from external_eval import run_external_eval as runner

    monkeypatch.setattr(ts, "TELEMETRY_DIR", tmp_path)
    monkeypatch.setattr(ts, "TELEMETRY_FILE", tmp_path / "telemetry.jsonl")

    out_csv = tmp_path / "agency_no_tools.csv"
    rc = runner.run(
        [
            "--target", "mock_echo",
            "--suite", "agency_social",
            "--no-gateway-analyze",
            "--output-csv", str(out_csv),
        ]
    )
    assert rc == 2  # no compatible cases → non-zero
    assert not out_csv.exists()


def test_run_external_eval_unknown_target(tmp_path):
    from external_eval import run_external_eval as runner

    rc = runner.run(
        [
            "--target", "does_not_exist",
            "--suite", "prompt_injection",
            "--no-gateway-analyze",
            "--max-attacks", "1",
            "--output-csv", str(tmp_path / "x.csv"),
        ]
    )
    assert rc == 2
