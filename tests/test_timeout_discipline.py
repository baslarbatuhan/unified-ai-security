"""tests/test_timeout_discipline.py
=====================================
Hafta 11 — timeout discipline + fail-CLOSED + LLMJudge breaker + telemetry.

Covers four layers:
  1. `configs/timeout_loader.py` helpers (on_timeout_policy, policy_risk_score,
     llm_judge_budget_ms) — pure-fn, no I/O.
  2. `FusionEngine._run_parallel` per-module fail-CLOSED behaviour — when a
     submitted future raises/times out, the synthesised ModuleRisk reflects
     the configured policy (block→1.0 / sanitize→0.5 / allow→0.0) and the
     evidence carries the breached budget.
  3. `rag_guard.llm_judge.LLMJudge` — on _call_ollama failure, `analyze`
     emits a structured ErrorEvent with `where="rag_guard.llm_judge.analyze"`
     and classifies TimeoutError vs CircuitOpenError vs other.
  4. CircuitBreaker integration — 3 consecutive failures open the circuit;
     subsequent calls short-circuit with a CircuitOpenError surfaced as
     `error="circuit_open: ..."` in the JudgeResult.
"""
from __future__ import annotations

from concurrent.futures import Future
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from configs.timeout_loader import (
    llm_judge_budget_ms,
    load_timeout_profile,
    on_timeout_policy,
    policy_risk_score,
    module_budget_ms,
)
from fusion_gateway import engine as eng
from fusion_gateway.engine import FusionEngine, ModuleRisk


# ---------------------------------------------------------------------------
# 1) Loader helpers
# ---------------------------------------------------------------------------
class TestLoaderHelpers:
    def test_on_timeout_policy_returns_value_from_profile(self) -> None:
        profile = {"on_timeout": {"prompt_guard": "sanitize", "rag_guard": "block"}}
        assert on_timeout_policy(profile, "prompt_guard") == "sanitize"
        assert on_timeout_policy(profile, "rag_guard") == "block"

    def test_on_timeout_policy_defaults_to_block_when_missing(self) -> None:
        """Fail-CLOSED by default — never silently fail-open."""
        assert on_timeout_policy({}, "anything") == "block"
        assert on_timeout_policy({"on_timeout": {}}, "missing") == "block"

    def test_on_timeout_policy_rejects_unknown_value(self) -> None:
        """Misconfig (typo) should not crash; reverts to default."""
        profile = {"on_timeout": {"prompt_guard": "explode"}}
        assert on_timeout_policy(profile, "prompt_guard") == "block"

    def test_on_timeout_policy_respects_custom_default(self) -> None:
        assert on_timeout_policy({}, "x", default="sanitize") == "sanitize"

    def test_policy_risk_score_mapping(self) -> None:
        assert policy_risk_score("allow") == 0.0
        assert policy_risk_score("sanitize") == 0.5
        assert policy_risk_score("block") == 1.0

    def test_policy_risk_score_unknown_is_fail_closed(self) -> None:
        """Unknown policy → 1.0 (max risk). Never silently fail-open."""
        assert policy_risk_score("whatever") == 1.0
        assert policy_risk_score("") == 1.0

    def test_llm_judge_budget_ms_reads_field(self) -> None:
        profile = {"llm_judge_ms": 12345}
        assert llm_judge_budget_ms(profile) == 12345

    def test_llm_judge_budget_ms_falls_back_to_default(self) -> None:
        assert llm_judge_budget_ms({}, default_ms=9999) == 9999

    def test_llm_judge_budget_ms_handles_garbage(self) -> None:
        assert llm_judge_budget_ms({"llm_judge_ms": "not-an-int"}, default_ms=5000) == 5000

    def test_standard_profile_loads_with_hafta11_fields(self) -> None:
        """timeout_config.yaml ships with the new fields populated."""
        profile = load_timeout_profile("standard")
        assert "llm_judge_ms" in profile
        assert "on_timeout" in profile
        # llm_judge sub-budget must fit inside rag_guard's overall budget.
        rag_ms = module_budget_ms(profile, "rag_guard")
        assert profile["llm_judge_ms"] <= rag_ms, (
            "llm_judge_ms must not exceed rag_guard_ms in standard profile"
        )


# ---------------------------------------------------------------------------
# 2) FusionEngine._run_parallel fail-CLOSED behaviour
# ---------------------------------------------------------------------------
def _completed_future_raising(exc: Exception) -> Future:
    """A Future already resolved to an exception — `.result(timeout=X)`
    re-raises immediately regardless of the timeout argument."""
    f: Future = Future()
    f.set_exception(exc)
    return f


def _completed_future_with(value: Any) -> Future:
    f: Future = Future()
    f.set_result(value)
    return f


class TestEngineWarmUp:
    """Hafta 15: opt-in warm-up via UAIS_WARM_UP_PIPELINES env var."""

    def test_warm_up_off_by_default(self, monkeypatch) -> None:
        """Test runs without the env var should never trigger warm-up.
        Otherwise pytest would pay the BGE-M3 load cost on every engine."""
        monkeypatch.delenv("UAIS_WARM_UP_PIPELINES", raising=False)
        calls = {"n": 0}
        monkeypatch.setattr(eng, "_get_prompt_pipeline",
                            lambda: calls.__setitem__("n", calls["n"] + 1))
        FusionEngine()
        assert calls["n"] == 0

    def test_warm_up_on_calls_pipeline_loader(self, monkeypatch) -> None:
        calls = {"n": 0}
        monkeypatch.setattr(eng, "_get_prompt_pipeline",
                            lambda: calls.__setitem__("n", calls["n"] + 1))
        monkeypatch.setenv("UAIS_WARM_UP_PIPELINES", "1")
        FusionEngine()
        assert calls["n"] == 1

    def test_warm_up_recognises_truthy_strings(self, monkeypatch) -> None:
        for value in ("true", "TRUE", "yes", "on"):
            calls = {"n": 0}
            monkeypatch.setattr(eng, "_get_prompt_pipeline",
                                lambda: calls.__setitem__("n", calls["n"] + 1))
            monkeypatch.setenv("UAIS_WARM_UP_PIPELINES", value)
            FusionEngine()
            assert calls["n"] == 1, f"failed for value={value!r}"

    def test_warm_up_ignores_other_values(self, monkeypatch) -> None:
        for value in ("0", "no", "off", "", "false", "anything-else"):
            calls = {"n": 0}
            monkeypatch.setattr(eng, "_get_prompt_pipeline",
                                lambda: calls.__setitem__("n", calls["n"] + 1))
            monkeypatch.setenv("UAIS_WARM_UP_PIPELINES", value)
            FusionEngine()
            assert calls["n"] == 0, f"warm-up triggered for value={value!r}"

    def test_warm_up_swallows_pipeline_exceptions(self, monkeypatch) -> None:
        """Engine construction must never fail because the pipeline
        loader threw — production startups can't crash on warm-up bugs."""
        def _boom():
            raise RuntimeError("cannot load model")
        monkeypatch.setattr(eng, "_get_prompt_pipeline", _boom)
        monkeypatch.setenv("UAIS_WARM_UP_PIPELINES", "1")
        # Should not raise.
        FusionEngine()


class TestRunParallelFailClosed:
    """Drive `_safe_result` directly by patching executor.submit so we don't
    need to actually wait on real workers. The inner closure pulls from the
    futures we hand it; the timeout policy comes from the engine's profile.
    """

    def test_timeout_emits_block_risk_when_policy_block(self, monkeypatch) -> None:
        # Stub all 3 evaluators so submit returns a future we control.
        engine = FusionEngine()
        # Force a clean policy state on the engine instance.
        engine._timeout_profile = {
            "modules": {"prompt_guard": 100, "rag_guard": 100, "output_agency": 100},
            "on_timeout": {"prompt_guard": "block", "rag_guard": "block", "output_agency": "block"},
        }

        # rag_guard future raises TimeoutError; prompt + agency return clean ModuleRisk.
        clean_prompt = ModuleRisk(module="prompt_guard", risk_score=0.0, confidence=1.0,
                                  decision="allow", evidence=["ok"], latency_ms=1)
        clean_agency = ModuleRisk(module="output_agency", risk_score=0.0, confidence=1.0,
                                  decision="allow", evidence=["ok"], latency_ms=1)

        from concurrent.futures import TimeoutError as FuturesTimeout

        original_submit = eng.ThreadPoolExecutor.submit

        def fake_submit(self, fn, *args, **kwargs):  # noqa: ANN001
            name = getattr(fn, "__name__", "")
            if name == "_evaluate_prompt_guard":
                return _completed_future_with(clean_prompt)
            if name == "_evaluate_rag_guard":
                return _completed_future_raising(FuturesTimeout("simulated"))
            if name == "_max_agency_risk":
                return _completed_future_with(clean_agency)
            return original_submit(self, fn, *args, **kwargs)

        monkeypatch.setattr(eng.ThreadPoolExecutor, "submit", fake_submit)
        enabled = {"prompt_guard": True, "rag_guard": True, "output_agency": True, "output_guard": True}
        p, r, a = engine._run_parallel(
            user_input="hi", retrieved_context=None, retrieved_docs=None,
            tool_call=None, user_id="u", role="basic", enabled=enabled,
        )
        assert r.module == "rag_guard"
        assert r.risk_score == 1.0          # block → 1.0
        assert r.decision == "block"
        assert any("TIMEOUT" in e for e in r.evidence)
        assert "rag_guard" in r.evidence[0]
        # Untouched modules pass through.
        assert p.risk_score == 0.0
        assert a.risk_score == 0.0

    def test_timeout_emits_sanitize_risk_when_policy_sanitize(self, monkeypatch) -> None:
        engine = FusionEngine()
        engine._timeout_profile = {
            "modules": {"prompt_guard": 100, "rag_guard": 100, "output_agency": 100},
            "on_timeout": {"prompt_guard": "sanitize", "rag_guard": "sanitize",
                           "output_agency": "sanitize"},
        }
        clean = ModuleRisk(module="prompt_guard", risk_score=0.0, confidence=1.0,
                           decision="allow", evidence=["ok"], latency_ms=1)

        def fake_submit(self, fn, *args, **kwargs):  # noqa: ANN001
            if getattr(fn, "__name__", "") == "_evaluate_prompt_guard":
                return _completed_future_raising(RuntimeError("kaboom"))
            return _completed_future_with(clean)

        monkeypatch.setattr(eng.ThreadPoolExecutor, "submit", fake_submit)
        enabled = {"prompt_guard": True, "rag_guard": True, "output_agency": True, "output_guard": True}
        p, _r, _a = engine._run_parallel(
            user_input="hi", retrieved_context=None, retrieved_docs=None,
            tool_call=None, user_id="u", role="basic", enabled=enabled,
        )
        assert p.module == "prompt_guard"
        assert p.risk_score == 0.5
        assert p.decision == "sanitize"
        assert any("RuntimeError" in e for e in p.evidence)

    def test_missing_profile_falls_back_to_block(self, monkeypatch) -> None:
        """Engine with no timeout_profile (config missing) still fail-CLOSED."""
        engine = FusionEngine()
        engine._timeout_profile = {}  # simulate missing config

        clean = ModuleRisk(module="output_agency", risk_score=0.0, confidence=1.0,
                           decision="allow", evidence=["ok"], latency_ms=1)

        def fake_submit(self, fn, *args, **kwargs):  # noqa: ANN001
            if getattr(fn, "__name__", "") == "_max_agency_risk":
                return _completed_future_raising(Exception("net"))
            return _completed_future_with(clean)

        monkeypatch.setattr(eng.ThreadPoolExecutor, "submit", fake_submit)
        enabled = {"prompt_guard": True, "rag_guard": True, "output_agency": True, "output_guard": True}
        _p, _r, a = engine._run_parallel(
            user_input="hi", retrieved_context=None, retrieved_docs=None,
            tool_call=None, user_id="u", role="basic", enabled=enabled,
        )
        assert a.decision == "block"  # default policy when missing
        assert a.risk_score == 1.0


# ---------------------------------------------------------------------------
# 3) LLMJudge ErrorEvent emission + classification
# ---------------------------------------------------------------------------
class _EventCapture:
    """In-test telemetry sink — replaces emit_telemetry so we can assert
    which events were produced without touching the real jsonl path."""

    def __init__(self) -> None:
        self.events: List[Any] = []

    def __call__(self, event: Any) -> None:
        self.events.append(event)


class TestLLMJudgeErrorTelemetry:
    def test_timeout_exception_emits_TimeoutError_event(self, monkeypatch) -> None:
        from rag_guard import llm_judge as lj

        # Fake `requests` so `_is_timeout_exc` recognises the exc type.
        class _FakeTimeout(Exception):
            pass

        class _FakeReqExceptions:
            Timeout = _FakeTimeout

        class _FakeRequests:
            exceptions = _FakeReqExceptions

        monkeypatch.setattr(lj, "requests", _FakeRequests)

        capture = _EventCapture()
        monkeypatch.setattr(lj, "emit_telemetry", capture)

        judge = lj.LLMJudge(timeout=1)
        # Force _call_ollama to raise the fake Timeout.
        monkeypatch.setattr(judge, "_call_ollama",
                            lambda *a, **kw: (_ for _ in ()).throw(_FakeTimeout("slow")))

        result = judge.analyze("doc", doc_id="d1", user_query="q")

        assert result.judge_score == 0.0
        assert result.error.startswith("timeout:"), f"unexpected error label: {result.error}"
        # Exactly one ErrorEvent emitted with our where/error_type.
        assert any(
            getattr(ev, "where", "") == "rag_guard.llm_judge.analyze"
            and getattr(ev, "error_type", "") == "TimeoutError"
            for ev in capture.events
        ), f"no matching ErrorEvent in {capture.events}"

    def test_generic_exception_emits_classname_event(self, monkeypatch) -> None:
        from rag_guard import llm_judge as lj

        capture = _EventCapture()
        monkeypatch.setattr(lj, "emit_telemetry", capture)

        judge = lj.LLMJudge(timeout=1)
        monkeypatch.setattr(judge, "_call_ollama",
                            lambda *a, **kw: (_ for _ in ()).throw(ValueError("nope")))

        result = judge.analyze("doc", doc_id="d1", user_query="q")
        assert "ValueError" in result.error or "nope" in result.error
        assert any(
            getattr(ev, "error_type", "") == "ValueError"
            for ev in capture.events
        )

    def test_circuit_open_exception_emits_CircuitOpenError_event(self, monkeypatch) -> None:
        from rag_guard import llm_judge as lj
        from utils.fallback_handler import CircuitOpenError

        capture = _EventCapture()
        monkeypatch.setattr(lj, "emit_telemetry", capture)

        judge = lj.LLMJudge(timeout=1)
        monkeypatch.setattr(
            judge, "_call_ollama",
            lambda *a, **kw: (_ for _ in ()).throw(
                CircuitOpenError("ollama_llm_judge", 0.0, 30.0)
            ),
        )
        result = judge.analyze("doc", doc_id="d1", user_query="q")
        assert result.error.startswith("circuit_open:"), result.error
        assert any(
            getattr(ev, "error_type", "") == "CircuitOpenError"
            for ev in capture.events
        )


# ---------------------------------------------------------------------------
# 4) CircuitBreaker integration on _call_ollama
# ---------------------------------------------------------------------------
class TestLLMJudgeCircuitBreaker:
    def test_three_failures_open_circuit_and_short_circuit_next_call(
        self, monkeypatch
    ) -> None:
        from rag_guard import llm_judge as lj
        from utils.fallback_handler import _REGISTRY, CircuitOpenError

        # Reset any breaker from previous tests to keep this isolated.
        _REGISTRY.clear()

        judge = lj.LLMJudge(timeout=1, breaker_name="ollama_llm_judge_test_3fail")

        # Patch requests so each `_call_ollama` raises a generic error,
        # tripping the breaker after `failure_threshold` consecutive fails.
        # `_call_ollama` builds the payload + calls requests.post → we
        # patch at the post level so the breaker's wrapper sees the raise.
        class _FakeReq:
            class exceptions:
                class Timeout(Exception):
                    pass

            @staticmethod
            def post(*args, **kwargs):  # noqa: ANN001
                raise _FakeReq.exceptions.Timeout("slow")

        monkeypatch.setattr(lj, "requests", _FakeReq)
        # Skip the model availability probe (uses requests.get).
        monkeypatch.setattr(judge, "_select_model", lambda: "stub-model")

        # Drive failures until the breaker opens. Default threshold is 3
        # (configs/service_limits.yaml). After that, _call_ollama should
        # raise CircuitOpenError without re-invoking the patched post.
        fails = 0
        for _ in range(3):
            try:
                judge._call_ollama("sys", "user")
            except Exception as e:
                fails += 1
                assert not isinstance(e, CircuitOpenError), (
                    "breaker opened before threshold"
                )
        assert fails == 3

        # 4th call must short-circuit.
        with pytest.raises(CircuitOpenError):
            judge._call_ollama("sys", "user")

    def test_analyze_with_open_circuit_returns_circuit_open_error_label(
        self, monkeypatch
    ) -> None:
        """End-to-end: when the breaker is OPEN, analyze() catches the
        CircuitOpenError and embeds it in the JudgeResult.error string."""
        from rag_guard import llm_judge as lj
        from utils.fallback_handler import _REGISTRY

        _REGISTRY.clear()
        judge = lj.LLMJudge(timeout=1, breaker_name="ollama_llm_judge_test_open")

        # Pre-open the breaker manually.
        assert judge._breaker is not None
        for _ in range(judge._breaker.failure_threshold):
            judge._breaker.record_failure()

        # Skip model probe.
        monkeypatch.setattr(judge, "_select_model", lambda: "stub-model")

        # Sink to capture telemetry.
        capture = _EventCapture()
        monkeypatch.setattr(lj, "emit_telemetry", capture)

        result = judge.analyze("doc body", doc_id="d-open", user_query="q?")
        assert result.judge_score == 0.0
        assert result.error.startswith("circuit_open:"), result.error
        # Confirm the structured event landed too.
        assert any(
            getattr(ev, "error_type", "") == "CircuitOpenError"
            for ev in capture.events
        )


# ---------------------------------------------------------------------------
# Backward-compat: legacy callers passing `timeout=int` to LLMJudge
# ---------------------------------------------------------------------------
class TestLLMJudgeLegacyTimeoutKwarg:
    def test_explicit_int_timeout_wins_over_profile(self) -> None:
        """Legacy callers that pass `timeout=N` keep their integer; the
        profile loader is bypassed so existing test fixtures don't shift."""
        from rag_guard.llm_judge import LLMJudge

        j = LLMJudge(timeout=7)
        assert j.timeout == 7

    def test_no_timeout_kwarg_pulls_from_profile_or_default(self) -> None:
        """When no `timeout=` is provided, judge takes whatever the
        active profile says (positive int) and never zero."""
        from rag_guard.llm_judge import LLMJudge

        j = LLMJudge()
        assert isinstance(j.timeout, int)
        assert j.timeout > 0
