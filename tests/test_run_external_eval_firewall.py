"""tests/test_run_external_eval_firewall.py
============================================
Sprint 11 — end-to-end firewall enforcement through the runner.

The headline guarantee: in firewall mode a `block` verdict means the adapter
is NEVER invoked. We prove it with a spy adapter that records every send().

Gateway analysis is stubbed (no torch / no network) so these stay unit-fast.
"""

from __future__ import annotations

import csv

import pytest

from schemas.target_schema import TargetConfig
from external_eval.base_adapter import ChatbotResponse


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class _StubGateway:
    """Returns a fixed decision for every analyze() call."""

    def __init__(self, decision: str):
        self.decision = decision
        self.calls = 0

    def analyze(self, request):  # noqa: ANN001 — duck-typed
        self.calls += 1
        return {
            "final_decision": self.decision,
            "decision": self.decision,
            "decision_band": self.decision,
            "fused_risk": 0.9 if self.decision == "block" else 0.1,
            "module_risks": [],
        }


class _SpyAdapter:
    """Records every prompt sent so the test can assert the firewall blocked it."""

    def __init__(self, target: TargetConfig):
        self.target = target
        self.sent: list[str] = []

    def send(self, prompt: str, *, session_context=None) -> ChatbotResponse:
        self.sent.append(prompt)
        return ChatbotResponse(text="REAL TARGET REPLY", ok=True, target_id=self.target.id)

    def close(self) -> None:
        pass


def _wire(monkeypatch, tmp_path, decision: str):
    """Patch telemetry, gateway and adapter; return the spy for assertions."""
    from schemas import telemetry_schema as ts
    from external_eval import run_external_eval as runner

    monkeypatch.setattr(ts, "TELEMETRY_DIR", tmp_path)
    monkeypatch.setattr(ts, "TELEMETRY_FILE", tmp_path / "telemetry.jsonl")

    monkeypatch.setattr(runner, "_build_gateway", lambda: _StubGateway(decision))

    spies: list[_SpyAdapter] = []

    def _fake_build_adapter(target):
        spy = _SpyAdapter(target)
        spies.append(spy)
        return spy

    monkeypatch.setattr(runner, "build_adapter", _fake_build_adapter)
    return runner, spies


def _read_rows(path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Firewall BLOCK — the prompt must never reach the target
# ---------------------------------------------------------------------------
def test_firewall_block_never_calls_adapter(tmp_path, monkeypatch):
    runner, spies = _wire(monkeypatch, tmp_path, decision="block")
    out_csv = tmp_path / "fw_block.csv"

    rc = runner.run([
        "--target", "mock_echo",
        "--suite", "prompt_injection",
        "--max-attacks", "2",
        "--firewall",
        "--output-csv", str(out_csv),
        "--run-id", "fw_block_run",
    ])
    assert rc == 0

    # The headline assertion: adapter.send was never called.
    assert spies and spies[0].sent == []

    rows = _read_rows(out_csv)
    assert len(rows) == 2
    for r in rows:
        assert r["mode"] == "firewall"
        assert r["firewall_action"] == "drop"
        assert r["forwarded_to_target"] == "0"
        assert r["adapter_state"] == "blocked_by_gateway_pre"
        assert r["adapter_ok"] == "0"           # synthetic response is not ok
        # gateway still ran → block verdict recorded for the audit trail
        assert r["gateway_decision"] == "block"


# ---------------------------------------------------------------------------
# Firewall ALLOW — the prompt is forwarded normally
# ---------------------------------------------------------------------------
def test_firewall_allow_forwards(tmp_path, monkeypatch):
    runner, spies = _wire(monkeypatch, tmp_path, decision="allow")
    out_csv = tmp_path / "fw_allow.csv"

    rc = runner.run([
        "--target", "mock_echo",
        "--suite", "prompt_injection",
        "--max-attacks", "2",
        "--firewall",
        "--output-csv", str(out_csv),
        "--run-id", "fw_allow_run",
    ])
    assert rc == 0

    assert spies and len(spies[0].sent) == 2  # both prompts forwarded

    rows = _read_rows(out_csv)
    for r in rows:
        assert r["mode"] == "firewall"
        assert r["firewall_action"] == "forward"
        assert r["forwarded_to_target"] == "1"
        assert r["adapter_state"] == "ok"
        assert r["adapter_ok"] == "1"


# ---------------------------------------------------------------------------
# Fail-closed — firewall mode without a gateway must abort
# ---------------------------------------------------------------------------
def test_firewall_without_gateway_fails_closed(tmp_path, monkeypatch):
    from schemas import telemetry_schema as ts
    from external_eval import run_external_eval as runner

    monkeypatch.setattr(ts, "TELEMETRY_DIR", tmp_path)
    monkeypatch.setattr(ts, "TELEMETRY_FILE", tmp_path / "telemetry.jsonl")

    out_csv = tmp_path / "fw_failclosed.csv"
    rc = runner.run([
        "--target", "mock_echo",
        "--suite", "prompt_injection",
        "--max-attacks", "1",
        "--firewall",
        "--no-gateway-analyze",   # no verdict source → nothing to enforce
        "--output-csv", str(out_csv),
    ])
    assert rc == 2
    assert not out_csv.exists()
