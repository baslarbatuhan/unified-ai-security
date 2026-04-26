"""tests/test_phase1b_alpha.py
================================
Phase 1B-α unit tests — rate limiter, circuit breaker, alert rules.
No LLM, no network, no sleeps > 50ms.
"""

from __future__ import annotations

import time

import pytest


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------
def _limiter_cfg(rpm=60, burst=5, mul=1.0):
    return {
        "rate_limit": {
            "default": {
                "requests_per_minute": rpm,
                "burst": burst,
                "window_seconds": 60,
                "burst_multiplier": mul,
            },
            "external_eval": {
                "requests_per_minute": 30,
                "burst": 2,
                "window_seconds": 60,
                "burst_multiplier": 1.0,
            },
        }
    }


def test_rate_limiter_allows_burst_then_denies():
    from utils.rate_limiter import RateLimiter

    rl = RateLimiter(_limiter_cfg(rpm=60, burst=5, mul=1.0))
    results = [rl.acquire("alice") for _ in range(5)]
    assert all(r.allowed for r in results), "first 5 calls fit capacity"

    deny = rl.acquire("alice")
    assert not deny.allowed
    assert deny.retry_after_sec > 0
    assert "rate_limit_exceeded" in deny.reason


def test_rate_limiter_refills_over_time(monkeypatch):
    from utils import rate_limiter as rl_mod
    from utils.rate_limiter import RateLimiter

    fake_now = {"t": 1000.0}
    monkeypatch.setattr(rl_mod.time, "monotonic", lambda: fake_now["t"])

    rl = RateLimiter(_limiter_cfg(rpm=60, burst=2, mul=1.0))  # 1 token/sec, cap 2
    assert rl.acquire("bob").allowed
    assert rl.acquire("bob").allowed
    assert not rl.acquire("bob").allowed

    fake_now["t"] += 1.5  # 1.5 tokens refilled → one call allowed
    assert rl.acquire("bob").allowed
    assert not rl.acquire("bob").allowed


def test_rate_limiter_partitions_by_identity():
    from utils.rate_limiter import RateLimiter

    rl = RateLimiter(_limiter_cfg(rpm=60, burst=2, mul=1.0))
    assert rl.acquire("alice").allowed
    assert rl.acquire("alice").allowed
    assert not rl.acquire("alice").allowed
    # Different identity gets its own bucket.
    assert rl.acquire("bob").allowed


def test_rate_limiter_partitions_by_tier():
    from utils.rate_limiter import RateLimiter

    rl = RateLimiter(_limiter_cfg(rpm=60, burst=2, mul=1.0))
    rl.acquire("alice", tier="default")
    rl.acquire("alice", tier="default")
    assert not rl.acquire("alice", tier="default").allowed
    # external_eval tier is its own bucket for the same identity.
    assert rl.acquire("alice", tier="external_eval").allowed


def test_rate_limiter_unknown_tier_falls_back_to_default():
    from utils.rate_limiter import RateLimiter

    rl = RateLimiter(_limiter_cfg(rpm=60, burst=3, mul=1.0))
    r = rl.acquire("x", tier="nonexistent")
    assert r.allowed
    assert r.tier == "default"


def test_rate_limiter_headers_on_deny():
    from utils.rate_limiter import RateLimiter

    rl = RateLimiter(_limiter_cfg(rpm=60, burst=1, mul=1.0))
    rl.acquire("x")
    deny = rl.acquire("x")
    h = deny.to_headers()
    assert "Retry-After" in h
    assert int(h["Retry-After"]) >= 1
    assert h["X-RateLimit-Tier"] == "default"


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------
def _breaker_cfg(ft=3, cd=0.05, hk=2):
    return {
        "circuit_breaker": {
            "failure_threshold": ft,
            "open_cooldown_seconds": cd,
            "half_open_success_threshold": hk,
        }
    }


def test_breaker_opens_after_threshold_failures():
    from utils.fallback_handler import CircuitBreaker, CircuitState, CircuitOpenError

    br = CircuitBreaker("test", config=_breaker_cfg(ft=3, cd=10))

    def _boom():
        raise RuntimeError("down")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            br.call(_boom)
    assert br.state == CircuitState.OPEN
    # Further calls short-circuit.
    with pytest.raises(CircuitOpenError):
        br.call(_boom)


def test_breaker_uses_fallback_when_open():
    from utils.fallback_handler import CircuitBreaker

    br = CircuitBreaker("test", config=_breaker_cfg(ft=2, cd=10))

    def _boom():
        raise RuntimeError("down")

    def _fallback():
        return "cached"

    for _ in range(2):
        br.call(_boom, fallback=_fallback)
    # Breaker now OPEN — fallback still short-circuits cleanly.
    assert br.call(_boom, fallback=_fallback) == "cached"
    assert br.stats().total_short_circuits >= 1


def test_breaker_half_open_probes_and_closes():
    from utils.fallback_handler import CircuitBreaker, CircuitState

    br = CircuitBreaker("test", config=_breaker_cfg(ft=2, cd=0.05, hk=2))

    def _boom():
        raise RuntimeError("down")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            br.call(_boom)
    assert br.state == CircuitState.OPEN

    time.sleep(0.07)  # cooldown elapses
    assert br.allow()  # flips to HALF_OPEN
    assert br.state == CircuitState.HALF_OPEN

    br.record_success()
    br.record_success()
    assert br.state == CircuitState.CLOSED


def test_breaker_half_open_failure_reopens():
    from utils.fallback_handler import CircuitBreaker, CircuitState

    br = CircuitBreaker("test", config=_breaker_cfg(ft=2, cd=0.05, hk=2))

    def _boom():
        raise RuntimeError("down")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            br.call(_boom)

    time.sleep(0.07)
    br.allow()  # → HALF_OPEN
    assert br.state == CircuitState.HALF_OPEN

    # Any failure during probe returns to OPEN.
    with pytest.raises(RuntimeError):
        br.call(_boom)
    assert br.state == CircuitState.OPEN


def test_breaker_registry_returns_same_instance():
    from utils.fallback_handler import get_breaker, reset_registry

    reset_registry()
    a = get_breaker("llm_judge")
    b = get_breaker("llm_judge")
    c = get_breaker("embedding")
    assert a is b
    assert a is not c
    reset_registry()


# ---------------------------------------------------------------------------
# Alert rules
# ---------------------------------------------------------------------------
def test_alert_rules_yaml_loads():
    from monitoring.alert_rules import load_rules

    rules = load_rules()
    assert len(rules) >= 5
    ids = [r.get("id") for r in rules]
    assert "high_error_rate" in ids
    assert "elevated_bypass_rate" in ids


def test_alert_rules_fire_on_high_error_rate():
    from monitoring.alert_rules import evaluate

    snapshot = {
        "total_requests": 100,
        "error_rate": 0.08,
        "block_rate": 0.1,
        "attack_bypass_rate": 0.0,
        "p95_latency_ms": 500,
    }
    alerts = evaluate(snapshot)
    ids = [a.rule_id for a in alerts]
    assert "high_error_rate" in ids
    fired = next(a for a in alerts if a.rule_id == "high_error_rate")
    assert fired.severity == "critical"
    assert "8" in fired.message or "0.08" in fired.message or "%" in fired.message


def test_alert_rules_suppress_on_low_traffic():
    """total_requests < 20 guard — rules requiring volume must not fire."""
    from monitoring.alert_rules import evaluate

    snapshot = {
        "total_requests": 5,
        "error_rate": 0.60,  # 3/5 errors — tiny sample
        "attack_bypass_rate": 0.40,
        "block_rate": 0.9,
    }
    alerts = evaluate(snapshot)
    assert not any(a.rule_id == "high_error_rate" for a in alerts)
    assert not any(a.rule_id == "elevated_bypass_rate" for a in alerts)


def test_alert_rules_missing_metric_does_not_fire():
    from monitoring.alert_rules import evaluate

    # No metrics at all — nothing should fire.
    assert evaluate({}) == []


def test_alert_rules_custom_rules_override():
    from monitoring.alert_rules import evaluate, Alert

    custom = [
        {"id": "p95_warn", "severity": "warn",
         "message": "p95={p95_latency_ms}", "when": [
             {"metric": "p95_latency_ms", "op": "gt", "threshold": 100}
         ]}
    ]
    alerts = evaluate({"p95_latency_ms": 150}, rules=custom)
    assert len(alerts) == 1
    assert isinstance(alerts[0], Alert)
    assert alerts[0].rule_id == "p95_warn"
    assert "150" in alerts[0].message


def test_build_snapshot_from_events_counts_decisions():
    from monitoring.alert_rules import build_snapshot_from_events

    events = [
        {"kind": "fusion_decision", "decision": "allow",  "latency_ms_total": 100,
         "prompt_score": 0.1, "rag_score": 0.0, "agency_score": 0.0, "output_score": 0.0},
        {"kind": "fusion_decision", "decision": "block",  "latency_ms_total": 200,
         "prompt_score": 0.9, "rag_score": 0.0, "agency_score": 0.0, "output_score": 0.0},
        {"kind": "fusion_decision", "decision": "sanitize","latency_ms_total": 150,
         "prompt_score": 0.5, "rag_score": 0.0, "agency_score": 0.0, "output_score": 0.0},
        # A bypass: module flagged ≥0.60 but fusion said allow
        {"kind": "fusion_decision", "decision": "allow",  "latency_ms_total": 300,
         "prompt_score": 0.7, "rag_score": 0.0, "agency_score": 0.0, "output_score": 0.0},
        {"kind": "module_result", "module": "rag_guard", "risk_score": 0.0,
         "confidence": 1.0, "decision": "allow", "latency_ms": 10, "evidence": ["ok"]},
        {"kind": "error", "where": "fusion", "error_type": "TimeoutError", "message": "x"},
    ]
    snap = build_snapshot_from_events(events)
    assert snap["total_requests"] == 4
    assert snap["block_rate"] == pytest.approx(0.25)
    assert snap["sanitize_rate"] == pytest.approx(0.25)
    assert snap["allow_rate"] == pytest.approx(0.50)
    assert snap["avg_latency_ms"] == pytest.approx((100 + 200 + 150 + 300) / 4)
    assert snap["p95_latency_ms"] >= 250
    # 2 blockable (scores ≥ 0.60: block=0.9 and bypass=0.7); 1 allowed → 0.5
    assert snap["attack_bypass_rate"] == pytest.approx(0.5)
    assert snap["module_rag_guard_avg_latency_ms"] == pytest.approx(10.0)
    assert snap["error_rate"] == pytest.approx(1 / 4)


def test_alert_rules_invalid_op_raises():
    from monitoring.alert_rules import evaluate

    bad = [{"id": "x", "severity": "info", "message": "", "when": [
        {"metric": "a", "op": "bogus", "threshold": 1}]}]
    with pytest.raises(KeyError):
        evaluate({"a": 5}, rules=bad)
