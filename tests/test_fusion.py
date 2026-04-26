"""tests/test_fusion.py
Regression tests for the FusionEngine — pure-logic coverage with stubbed
module evaluators. No LLM calls, no embeddings, no I/O.

We verify:
    * `_threshold_decision` — score buckets to allow/sanitize/flag/block
    * weighted-sum fusion respects the configured weights
    * elevated/critical override lifts fused above weighted_sum when a
      single module is highly confident
    * disabling a module renormalises weights instead of zero-padding
    * to_dict round-trips the response shape
"""
from __future__ import annotations

import pytest

import fusion_gateway.engine as eng
from fusion_gateway.engine import (
    FusionEngine,
    ModuleRisk,
    _threshold_decision,
)


# ---------------------------------------------------------------------------
# _threshold_decision (pure)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "score,expected",
    [
        (0.00, "allow"),
        (eng.DEFAULT_THRESHOLDS["allow"] - 1e-6, "allow"),
        (eng.DEFAULT_THRESHOLDS["allow"], "sanitize"),
        (eng.DEFAULT_THRESHOLDS["sanitize"] - 1e-6, "sanitize"),
        (eng.DEFAULT_THRESHOLDS["sanitize"], "flag"),
        (eng.DEFAULT_THRESHOLDS["block"] - 1e-6, "flag"),
        (eng.DEFAULT_THRESHOLDS["block"], "block"),
        (1.00, "block"),
    ],
)
def test_threshold_decision_buckets(score, expected):
    assert _threshold_decision(score) == expected


# ---------------------------------------------------------------------------
# Stub helpers — patch the three evaluators so analyze() never touches LLMs
# ---------------------------------------------------------------------------
def _stub_evaluators(monkeypatch, *, p=0.0, r=0.0, a=0.0):
    """Patch the module evaluators to return deterministic ModuleRisks."""
    monkeypatch.setattr(
        eng, "_evaluate_prompt_guard",
        lambda user_input: ModuleRisk(module="prompt_guard", risk_score=p, confidence=1.0, latency_ms=1),
    )
    monkeypatch.setattr(
        eng, "_evaluate_rag_guard",
        lambda **kw: ModuleRisk(module="rag_guard", risk_score=r, confidence=1.0, latency_ms=1),
    )
    monkeypatch.setattr(
        eng, "_max_agency_risk",
        lambda *a_, **kw: ModuleRisk(module="output_agency", risk_score=a, confidence=1.0, latency_ms=1),
    )
    # Force all-modules-enabled regardless of YAML.
    monkeypatch.setattr(eng, "_module_enabled_flags", lambda: {
        "prompt_guard": True, "rag_guard": True, "output_agency": True, "output_guard": True,
    })


def _engine_serial():
    """Engine in serial mode so monkeypatched evaluators are used predictably."""
    return FusionEngine(parallel=False)


# ---------------------------------------------------------------------------
# D4 — per-request overrides actually steer the decision
# ---------------------------------------------------------------------------
def test_overrides_thresholds_change_decision(monkeypatch):
    """Same module scores, two different threshold sets → two different decisions."""
    _stub_evaluators(monkeypatch, p=0.55, r=0.0, a=0.0)
    e = _engine_serial()
    strict = e.analyze(
        user_input="x",
        overrides={"thresholds": {"allow": 0.05, "sanitize": 0.10, "block": 0.15}},
    )
    relaxed = e.analyze(
        user_input="x",
        overrides={"thresholds": {"allow": 0.90, "sanitize": 0.95, "block": 0.99}},
    )
    assert strict.final_decision == "block"
    assert relaxed.final_decision == "allow"


def test_overrides_weights_renormalise(monkeypatch):
    """Heavier weight on the high-scoring module raises fused_risk."""
    _stub_evaluators(monkeypatch, p=0.90, r=0.10, a=0.10)
    e = _engine_serial()
    balanced = e.analyze(
        user_input="x",
        overrides={"weights": {"prompt_guard": 0.34, "rag_guard": 0.33, "output_agency": 0.33}},
    )
    prompt_heavy = e.analyze(
        user_input="x",
        overrides={"weights": {"prompt_guard": 0.80, "rag_guard": 0.10, "output_agency": 0.10}},
    )
    assert prompt_heavy.fused_risk >= balanced.fused_risk


def test_overrides_modules_enabled_skips_evaluator(monkeypatch):
    """modules_enabled=False keeps the evaluator off the call path entirely."""
    calls = {"rag": 0}

    def counting_rag(**kw):
        calls["rag"] += 1
        return ModuleRisk(module="rag_guard", risk_score=0.99, confidence=1.0, latency_ms=1)

    _stub_evaluators(monkeypatch, p=0.0, r=0.0, a=0.0)
    # Replace rag evaluator AFTER _stub_evaluators so our counter sticks.
    monkeypatch.setattr(eng, "_evaluate_rag_guard", counting_rag)

    e = _engine_serial()
    e.analyze(user_input="x", overrides={"modules_enabled": {"rag_guard": False}})
    assert calls["rag"] == 0, "rag_guard evaluator must not run when override disables it"


# ---------------------------------------------------------------------------
# Weighted-sum fusion
# ---------------------------------------------------------------------------
def test_low_scores_yield_allow(monkeypatch):
    _stub_evaluators(monkeypatch, p=0.05, r=0.05, a=0.05)
    resp = _engine_serial().analyze(user_input="hi")
    assert resp.final_decision == "allow"
    assert resp.fused_risk < eng.DEFAULT_THRESHOLDS["allow"]


def test_weighted_sum_matches_formula(monkeypatch):
    _stub_evaluators(monkeypatch, p=0.10, r=0.20, a=0.30)
    eng_ = _engine_serial()
    w = eng_.weights
    resp = eng_.analyze(user_input="x")
    expected = (w["prompt_guard"] * 0.10 + w["rag_guard"] * 0.20 + w["output_agency"] * 0.30) / (
        w["prompt_guard"] + w["rag_guard"] + w["output_agency"]
    )
    # Override may lift the score, but with all modules <0.60 the override is dormant.
    assert resp.fused_risk == pytest.approx(round(expected, 4), abs=1e-3)


def test_critical_single_module_overrides_dilution(monkeypatch):
    """One module at 0.95, others at 0 — fused should reflect critical severity."""
    _stub_evaluators(monkeypatch, p=0.95, r=0.0, a=0.0)
    resp = _engine_serial().analyze(user_input="x")
    # critical_threshold=0.85, critical_multiplier=0.90 → fused >= 0.95 * 0.90 = 0.855
    assert resp.fused_risk >= 0.85
    assert resp.final_decision == "block"


def test_elevated_single_module_lifts_to_flag(monkeypatch):
    """One module at 0.65 (elevated), others at 0 — without override would be ≈0.20."""
    _stub_evaluators(monkeypatch, p=0.65, r=0.0, a=0.0)
    resp = _engine_serial().analyze(user_input="x")
    # elevated_multiplier 0.85 → fused >= 0.65 * 0.85 = 0.5525, decision sanitize/flag (>= sanitize threshold)
    assert resp.fused_risk >= 0.55
    assert resp.final_decision in ("sanitize", "flag", "block")


# ---------------------------------------------------------------------------
# Module disable renormalisation
# ---------------------------------------------------------------------------
def test_disabled_module_renormalises(monkeypatch):
    """If RAG is disabled, prompt+agency weights should be renormalised so a
    high prompt score still drives the fused risk — not diluted by RAG=0."""
    monkeypatch.setattr(eng, "_module_enabled_flags", lambda: {
        "prompt_guard": True, "rag_guard": False, "output_agency": True, "output_guard": True,
    })
    monkeypatch.setattr(
        eng, "_evaluate_prompt_guard",
        lambda user_input: ModuleRisk(module="prompt_guard", risk_score=0.40, latency_ms=1),
    )
    # rag_guard evaluator should NOT be called when disabled — wire it to fail.
    def _boom(**kw):
        raise AssertionError("rag_guard must not be evaluated when disabled")
    monkeypatch.setattr(eng, "_evaluate_rag_guard", _boom)
    monkeypatch.setattr(
        eng, "_max_agency_risk",
        lambda *a_, **kw: ModuleRisk(module="output_agency", risk_score=0.0, latency_ms=1),
    )

    resp = _engine_serial().analyze(user_input="x")
    w = eng.DEFAULT_WEIGHTS
    # Renormalised: 0.40 * w_prompt / (w_prompt + w_agency)
    expected = (w["prompt_guard"] * 0.40) / (w["prompt_guard"] + w["output_agency"])
    assert resp.fused_risk == pytest.approx(round(expected, 4), abs=1e-3)


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------
def test_response_to_dict_shape(monkeypatch):
    _stub_evaluators(monkeypatch, p=0.10)
    resp = _engine_serial().analyze(user_input="x")
    d = resp.to_dict()
    assert set(d.keys()) == {"final_decision", "fused_risk", "module_risks", "latency_ms"}
    assert len(d["module_risks"]) == 3
    names = {m["module"] for m in d["module_risks"]}
    assert names == {"prompt_guard", "rag_guard", "output_agency"}


def test_module_risks_carry_evidence_and_latency(monkeypatch):
    monkeypatch.setattr(eng, "_module_enabled_flags", lambda: {
        "prompt_guard": True, "rag_guard": True, "output_agency": True, "output_guard": True,
    })
    monkeypatch.setattr(
        eng, "_evaluate_prompt_guard",
        lambda user_input: ModuleRisk(
            module="prompt_guard", risk_score=0.7, latency_ms=42, evidence=["pattern:ignore"],
        ),
    )
    monkeypatch.setattr(
        eng, "_evaluate_rag_guard",
        lambda **kw: ModuleRisk(module="rag_guard", risk_score=0.0, latency_ms=3),
    )
    monkeypatch.setattr(
        eng, "_max_agency_risk",
        lambda *a_, **kw: ModuleRisk(module="output_agency", risk_score=0.0, latency_ms=1),
    )
    resp = _engine_serial().analyze(user_input="x")
    pg = next(m for m in resp.module_risks if m["module"] == "prompt_guard")
    assert pg["latency_ms"] == 42
    assert "pattern:ignore" in pg["evidence"]
