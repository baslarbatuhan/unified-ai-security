"""tests/test_phase2b_analyze_output.py
==========================================
Post-LLM endpoint coverage.

Goals:
    * /analyze-output wires through to engine.analyze_with_output().
    * Missing `model_output` is a 422 — we never silently fall back.
    * Telemetry carries a non-zero output_score so dashboards can surface it.
    * external_eval rename (attack_success → gateway_miss) actually took.

We stub FusionEngine so the test stays hermetic (no ML models, no Ollama).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Stub engine — mirrors the fields SecurityGateway reads.
# ---------------------------------------------------------------------------
class _StubResponse:
    def __init__(self, decision, fused, module_risks, latency_ms):
        self.final_decision = decision
        self.fused_risk = fused
        self.module_risks = module_risks
        self.latency_ms = latency_ms


class _StubEngine:
    """Records whether analyze / analyze_with_output was called, returns
    deterministic scores the caller can assert on."""

    def __init__(self):
        self.calls: List[str] = []

    def analyze(self, **kwargs) -> _StubResponse:
        self.calls.append("analyze")
        return _StubResponse(
            decision="allow",
            fused=0.10,
            module_risks=[
                {"module": "prompt_guard", "risk_score": 0.10, "confidence": 1.0,
                 "decision": "allow", "evidence": ["benign"], "latency_ms": 1},
                {"module": "rag_guard", "risk_score": 0.05, "confidence": 1.0,
                 "decision": "allow", "evidence": [], "latency_ms": 0},
                {"module": "output_agency", "risk_score": 0.0, "confidence": 1.0,
                 "decision": "allow", "evidence": [], "latency_ms": 0},
            ],
            latency_ms=2,
        )

    def analyze_with_output(self, **kwargs) -> _StubResponse:
        self.calls.append("analyze_with_output")
        return _StubResponse(
            decision="block",
            fused=0.82,
            module_risks=[
                {"module": "prompt_guard", "risk_score": 0.10, "confidence": 1.0,
                 "decision": "allow", "evidence": [], "latency_ms": 1},
                {"module": "rag_guard", "risk_score": 0.0, "confidence": 1.0,
                 "decision": "allow", "evidence": [], "latency_ms": 0},
                {"module": "output_agency", "risk_score": 0.0, "confidence": 1.0,
                 "decision": "allow", "evidence": [], "latency_ms": 0},
                {"module": "output_guard", "risk_score": 0.88, "confidence": 0.9,
                 "decision": "block", "evidence": ["pii_leak"], "latency_ms": 3},
            ],
            latency_ms=5,
        )


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Spin up /analyze-output alone — no model warmup, no real engine."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    # Redirect telemetry to a scratch file so we can inspect writes.
    from schemas import telemetry_schema as ts
    monkeypatch.setattr(ts, "TELEMETRY_DIR", tmp_path)
    monkeypatch.setattr(ts, "TELEMETRY_FILE", tmp_path / "system_telemetry.jsonl")

    # Bypass api_main's startup (models). Build a minimal app that mounts
    # /analyze-output against our stubbed gateway.
    from api.security_gateway import SecurityGateway
    from api import api_main  # noqa: F401 — ensures module is import-clean
    from api.api_main import (
        AnalyzeWithOutputRequestModel, AnalyzeResponseModel, analyze_output,
    )

    stub = _StubEngine()
    gateway = SecurityGateway(engine=stub)
    monkeypatch.setattr("api.api_main.gateway", gateway)

    app = FastAPI()
    app.post("/analyze-output", response_model=AnalyzeResponseModel)(analyze_output)
    app.post("/analyze")(__import__("api.api_main", fromlist=["analyze"]).analyze)
    return TestClient(app), stub, tmp_path / "system_telemetry.jsonl"


# ---------------------------------------------------------------------------
# /analyze-output
# ---------------------------------------------------------------------------
def test_analyze_output_routes_to_with_output_method(client):
    c, stub, _ = client
    r = c.post("/analyze-output", json={
        "prompt": "hello",
        "model_output": "Here is my social security number: 123-45-6789",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["final_decision"] == "block"
    assert body["output_score"] == pytest.approx(0.88, rel=1e-3)
    # The input-side three scores survive — we're additive, not replacing.
    assert body["prompt_score"] == pytest.approx(0.10, rel=1e-3)
    assert stub.calls == ["analyze_with_output"]


def test_analyze_output_requires_model_output(client):
    c, _, _ = client
    # Missing model_output is a schema violation — pydantic 422.
    r = c.post("/analyze-output", json={"prompt": "hello"})
    assert r.status_code == 422


def test_analyze_output_emits_telemetry_with_output_score(client):
    c, _, telemetry_path = client
    r = c.post("/analyze-output", json={
        "prompt": "hi",
        "model_output": "leaked key sk-xxxx",
    })
    assert r.status_code == 200
    assert telemetry_path.exists(), "telemetry file should exist after call"
    lines = [l for l in telemetry_path.read_text().splitlines() if l.strip()]
    kinds = [__import__("json").loads(l)["kind"] for l in lines]
    # 1 request + 4 module_result (incl output_guard) + 1 fusion_decision.
    assert kinds.count("fusion_decision") == 1
    assert kinds.count("module_result") == 4
    # Fusion event must carry a non-zero output_score so the dashboard can
    # tell apart /analyze (output_score=0.0) from /analyze-output.
    fusion_line = next(l for l in lines if '"fusion_decision"' in l)
    assert '"output_score":0.88' in fusion_line or '"output_score": 0.88' in fusion_line


def test_plain_analyze_still_has_zero_output_score(client):
    c, stub, _ = client
    r = c.post("/analyze", json={"prompt": "hello"})
    assert r.status_code == 200
    assert r.json()["output_score"] == 0.0
    assert stub.calls == ["analyze"]  # did NOT call analyze_with_output


# ---------------------------------------------------------------------------
# external_eval rename — attack_success → gateway_miss
# ---------------------------------------------------------------------------
def test_external_eval_exports_gateway_miss_not_attack_success():
    from external_eval import run_external_eval as ree

    # The public field list the CSV writer uses must carry the new name.
    assert "gateway_miss" in ree._CSV_FIELDS
    assert "attack_success" not in ree._CSV_FIELDS
    # Classifier was renamed, old function is gone.
    assert hasattr(ree, "_classify_gateway_miss")
    assert not hasattr(ree, "_classify_attack_success")


def test_gateway_miss_classifier_semantics():
    from external_eval.run_external_eval import _classify_gateway_miss as cls
    assert cls("block", "allow") == 1          # expected block, got allow → miss
    assert cls("sanitize", "allow") == 1        # same for sanitize
    assert cls("block", "block") == 0           # caught → not a miss
    assert cls("allow", "allow") == 0           # benign-expected → not a miss
    assert cls("block", None) == 0              # gateway didn't run → abstain
