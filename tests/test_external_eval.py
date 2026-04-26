"""tests/test_external_eval.py
Smoke + behaviour tests for the external_eval harness.

We exercise:
    * MockAdapter — deterministic in-process target
    * adapter_factory.build_adapter dispatch
    * _classify_gateway_miss semantics (the proxy metric we renamed away
      from `attack_success`)
    * end-to-end run() against the mock target with `--no-gateway-analyze`
      and verify the CSV shape

The end-to-end test runs in milliseconds because the mock adapter never
makes network calls.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from external_eval.run_external_eval import _classify_gateway_miss, _CSV_FIELDS, run
from external_eval.mock_adapter import MockAdapter
from external_eval.adapter_factory import build_adapter
from external_eval.base_adapter import AdapterConfigError
from schemas.target_schema import TargetConfig


# ---------------------------------------------------------------------------
# MockAdapter
# ---------------------------------------------------------------------------
def test_mock_adapter_echoes_with_safe_preamble():
    target = TargetConfig(id="mock_t", name="t", type="mock", enabled=True)
    adapter = MockAdapter(target)
    resp = adapter.send("hello world")
    assert resp.ok
    assert "mock chatbot" in resp.text.lower()
    assert "hello world" in resp.text
    # Mock signals via metadata so callers can detect synthetic responses
    assert resp.metadata.get("mock") is True


def test_mock_adapter_rejects_non_mock_type():
    target = TargetConfig(id="x", name="x", type="api", enabled=True,
                          endpoint="https://example.test/")
    with pytest.raises(AdapterConfigError):
        MockAdapter(target)


def test_adapter_factory_dispatches_mock():
    target = TargetConfig(id="mock_t", name="t", type="mock", enabled=True)
    adapter = build_adapter(target)
    assert isinstance(adapter, MockAdapter)


# ---------------------------------------------------------------------------
# _classify_gateway_miss (pure)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("expected,decision,want", [
    # block/sanitize expected, gateway said allow → miss
    ("block",    "allow",    1),
    ("sanitize", "allow",    1),
    # gateway caught it → not a miss
    ("block",    "block",    0),
    ("block",    "sanitize", 0),
    ("sanitize", "block",    0),
    # gateway not run (None) → abstain
    ("block",    None,       0),
    ("sanitize", None,       0),
    # benign expected → never a miss regardless
    ("allow",    "allow",    0),
    ("allow",    "block",    0),
])
def test_classify_gateway_miss(expected, decision, want):
    assert _classify_gateway_miss(expected, decision) == want


# ---------------------------------------------------------------------------
# End-to-end run()
# ---------------------------------------------------------------------------
def test_run_against_mock_writes_well_formed_csv(tmp_path: Path):
    out = tmp_path / "results.csv"
    rc = run([
        "--target", "mock_echo",
        "--suite", "prompt_injection",
        "--max-attacks", "3",
        "--no-gateway-analyze",
        "--output-csv", str(out),
    ])
    assert rc == 0
    assert out.exists()

    with out.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    # Schema lock — fields must match _CSV_FIELDS exactly
    assert set(rows[0].keys()) == set(_CSV_FIELDS)

    # Mock adapter is always healthy and never errors
    for row in rows:
        assert row["adapter_ok"] == "1"
        assert row["adapter_error"] == ""
        assert int(row["response_chars"]) > 0
        assert row["target_id"] == "mock_echo"
        assert row["suite"] == "prompt_injection"
        # gateway disabled → no decision recorded, no miss claimed
        assert row["gateway_decision"] == ""
        assert row["gateway_miss"] == "0"


def test_run_unknown_target_returns_error(tmp_path: Path):
    out = tmp_path / "results.csv"
    rc = run([
        "--target", "no-such-target-xyz",
        "--max-attacks", "1",
        "--no-gateway-analyze",
        "--output-csv", str(out),
    ])
    assert rc == 2
    assert not out.exists()
