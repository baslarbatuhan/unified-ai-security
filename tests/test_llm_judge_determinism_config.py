"""
tests/test_llm_judge_determinism_config.py
==========================================
Fast unit-level assertions that LLMJudge sends deterministic sampling
options on the wire (temperature=0, seed=42, low top_p/top_k). These
settings underpin the Task 3 determinism claim in
reports/judge_determinism.md.

Does NOT call Ollama — it monkey-patches requests.post to intercept the
outbound payload.
"""

from __future__ import annotations

import pytest

from rag_guard.llm_judge import LLMJudge


class _FakeResp:
    status_code = 200
    text = ""

    def json(self):
        return {"message": {"content": '{"risk_score": 0.0, "reason": "probe"}'}}


@pytest.fixture()
def captured_payload(monkeypatch):
    """Intercept requests.post and capture the JSON payload."""
    import requests as _rq  # type: ignore
    captured: dict = {}

    def _patched_post(url, *args, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs.get("json") or (args[0] if args else None)
        return _FakeResp()

    monkeypatch.setattr(_rq, "post", _patched_post, raising=True)
    return captured


def test_judge_sends_temperature_zero(captured_payload):
    judge = LLMJudge()
    try:
        judge.analyze("probe text", doc_id="probe", user_query="probe")
    except Exception:
        # Parsing may fail on the canned response; we only care about the payload.
        pass
    options = (captured_payload.get("json") or {}).get("options") or {}
    assert options.get("temperature") == 0 or options.get("temperature") == 0.0, (
        f"judge request must use temperature=0, got {options.get('temperature')!r}"
    )


def test_judge_sends_fixed_seed(captured_payload):
    judge = LLMJudge()
    try:
        judge.analyze("probe text", doc_id="probe", user_query="probe")
    except Exception:
        pass
    options = (captured_payload.get("json") or {}).get("options") or {}
    assert options.get("seed") == 42, (
        f"judge request must use seed=42, got {options.get('seed')!r}"
    )


def test_judge_sends_greedy_sampling_options(captured_payload):
    """top_p and top_k should be tight so decoding is effectively greedy."""
    judge = LLMJudge()
    try:
        judge.analyze("probe text", doc_id="probe", user_query="probe")
    except Exception:
        pass
    options = (captured_payload.get("json") or {}).get("options") or {}
    # Accept either top_k=1 (strict greedy) or top_p<=0.1 (near-greedy nucleus).
    assert options.get("top_k") == 1 or options.get("top_p", 1.0) <= 0.1, (
        f"judge sampling is not greedy: options={options!r}"
    )
