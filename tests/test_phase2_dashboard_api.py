"""tests/test_phase2_dashboard_api.py
=======================================
Phase 2 unit tests — dashboard routes + rate-limit middleware.

Strategy: mount a minimal FastAPI app with only the dashboard router and
middleware so we never import api_main (which loads models on startup).
Telemetry/CSV paths are redirected to tmp_path so tests don't touch real
run artefacts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# App factory — isolated, no model warmup, no side effects.
# ---------------------------------------------------------------------------
def _make_app(limiter=None):
    from api.dashboard_routes import router
    from api.middleware import RateLimitMiddleware

    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, limiter=limiter)
    app.include_router(router)

    # A trivial echo route so we can exercise middleware without /analyze.
    @app.get("/ping")
    def ping():
        return {"ok": True}

    return app


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """Client with telemetry + CSV paths pointed into tmp_path."""
    from schemas import telemetry_schema as ts
    from api import dashboard_routes as dr
    from utils.fallback_handler import reset_registry
    from utils.rate_limiter import reset_default_limiter

    reset_registry()
    reset_default_limiter()

    tele = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr(ts, "TELEMETRY_DIR", tmp_path)
    monkeypatch.setattr(ts, "TELEMETRY_FILE", tele)
    monkeypatch.setattr(dr, "_RUNS_DIR", tmp_path)

    # Generous limiter so normal tests never hit 429 by accident.
    from utils.rate_limiter import RateLimiter
    limiter = RateLimiter({"rate_limit": {
        "default":       {"requests_per_minute": 6000, "burst": 200, "burst_multiplier": 1.0},
        "admin":         {"requests_per_minute": 6000, "burst": 200, "burst_multiplier": 1.0},
        "external_eval": {"requests_per_minute": 6000, "burst": 200, "burst_multiplier": 1.0},
    }})
    app = _make_app(limiter=limiter)
    client = TestClient(app)
    return client, tmp_path, tele


# ---------------------------------------------------------------------------
# /dashboard/summary + /dashboard/alerts
# ---------------------------------------------------------------------------
def _write_events(path: Path, events):
    with path.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def test_summary_empty_telemetry_returns_zeros(app_client):
    client, _, _ = app_client
    r = client.get("/dashboard/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["event_window"] == 0
    assert body["metrics"]["total_requests"] == 0
    assert body["alerts"] == []
    assert body["breakers"] == []


def test_summary_reflects_written_telemetry(app_client):
    client, _, tele = app_client
    events = [
        {"kind": "fusion_decision", "run_id": "r1", "decision": "allow",
         "fused_risk_score": 0.1, "latency_ms_total": 120,
         "prompt_score": 0.05, "rag_score": 0.0, "agency_score": 0.0, "output_score": 0.0,
         "evidence": ["ok"]},
        {"kind": "fusion_decision", "run_id": "r1", "decision": "block",
         "fused_risk_score": 0.9, "latency_ms_total": 200,
         "prompt_score": 0.92, "rag_score": 0.0, "agency_score": 0.0, "output_score": 0.0,
         "evidence": ["injection"]},
    ]
    _write_events(tele, events)

    body = client.get("/dashboard/summary").json()
    assert body["event_window"] == 2
    assert body["metrics"]["total_requests"] == 2
    assert body["metrics"]["block_rate"] == 0.5
    assert body["metrics"]["allow_rate"] == 0.5


def test_alerts_respect_min_severity(app_client):
    client, _, tele = app_client
    # 25 decisions with 10% bypass rate → elevated_bypass_rate (critical).
    events = []
    for i in range(25):
        is_bypass = i < 3  # 3/25 ≈ 12%
        events.append({
            "kind": "fusion_decision", "run_id": "r1",
            "decision": "allow" if is_bypass else "block",
            "fused_risk_score": 0.5, "latency_ms_total": 100,
            "prompt_score": 0.8 if i < 10 else 0.1,  # 10 blockable
            "rag_score": 0.0, "agency_score": 0.0, "output_score": 0.0,
            "evidence": [],
        })
    _write_events(tele, events)

    # Default min_severity=info → everything surfaces.
    all_alerts = client.get("/dashboard/alerts").json()
    ids = {a["rule_id"] for a in all_alerts["alerts"]}
    assert "elevated_bypass_rate" in ids

    # min_severity=critical → only critical survive.
    crit = client.get("/dashboard/alerts", params={"min_severity": "critical"}).json()
    assert all(a["severity"] == "critical" for a in crit["alerts"])
    assert crit["count"] >= 1


def test_alert_rules_endpoint_lists_config():
    # Doesn't need fixture — reads configs/alert_rules.yaml directly.
    app = _make_app()
    client = TestClient(app)
    body = client.get("/dashboard/alert-rules").json()
    assert body["count"] >= 5
    assert any(r["id"] == "high_error_rate" for r in body["rules"])


# ---------------------------------------------------------------------------
# /dashboard/recent-runs
# ---------------------------------------------------------------------------
def test_recent_runs_returns_newest_first_and_trims_evidence(app_client):
    client, _, tele = app_client
    events = [
        {"kind": "module_result", "run_id": "r1", "module": "prompt_guard",
         "risk_score": 0.1, "confidence": 1.0, "decision": "allow", "latency_ms": 5,
         "evidence": []},
    ]
    # 5 fusion decisions, each with 10 evidence lines (must be trimmed to 3).
    for i in range(5):
        events.append({
            "kind": "fusion_decision", "run_id": "r1", "case_id": f"c{i}",
            "decision": "allow", "fused_risk_score": 0.1 * i,
            "latency_ms_total": 100 + i, "prompt_score": 0.0, "rag_score": 0.0,
            "agency_score": 0.0, "output_score": 0.0,
            "evidence": [f"ev{j}" for j in range(10)],
        })
    _write_events(tele, events)

    body = client.get("/dashboard/recent-runs", params={"limit": 3}).json()
    assert body["count"] == 3
    assert body["decisions"][0]["case_id"] == "c4"  # newest first
    assert len(body["decisions"][0]["evidence"]) == 3


# ---------------------------------------------------------------------------
# /dashboard/circuit-breakers + /dashboard/rate-limits
# ---------------------------------------------------------------------------
def test_circuit_breakers_empty_then_populated(app_client):
    client, _, _ = app_client
    body = client.get("/dashboard/circuit-breakers").json()
    assert body == {"count": 0, "breakers": []}

    from utils.fallback_handler import get_breaker
    br = get_breaker("llm_judge")
    # Trigger one failure to make the stats non-trivial.
    try:
        br.call(lambda: (_ for _ in ()).throw(RuntimeError("x")))
    except RuntimeError:
        pass

    body = client.get("/dashboard/circuit-breakers").json()
    assert body["count"] == 1
    assert body["breakers"][0]["name"] == "llm_judge"
    assert body["breakers"][0]["total_failures"] == 1


def test_rate_limits_returns_snapshot(app_client):
    client, _, _ = app_client
    # The endpoint reads from default_limiter() — populate it directly so
    # this test doesn't depend on middleware wiring (which is tested
    # separately below).
    from utils.rate_limiter import default_limiter
    default_limiter().acquire("alice", tier="default")

    body = client.get("/dashboard/rate-limits").json()
    assert "buckets" in body
    assert any(k.startswith("default:") for k in body["buckets"].keys())


# ---------------------------------------------------------------------------
# CSV tailers
# ---------------------------------------------------------------------------
def _write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_output_guard_metrics_endpoint(app_client):
    client, runs_dir, _ = app_client
    path = runs_dir / "output_security_metrics.csv"
    rows = [
        {"run_id": "r", "case_id": f"c{i}", "target_id": "t",
         "score": 0.1 * i, "decision": "allow", "output_chars": 100,
         "latency_ms": 10, "flag_pii": 0, "flag_api_key": 0,
         "flag_unsafe_instruction": 0, "flag_downstream_injection": 0,
         "flag_redirect_to_unknown": 0, "evidence_top": ""}
        for i in range(5)
    ]
    _write_csv(path, rows, list(rows[0].keys()))

    body = client.get("/dashboard/metrics/output-guard", params={"limit": 3}).json()
    assert body["count"] == 3
    assert body["source"] == "output_security_metrics.csv"
    # Last 3 rows (case_id = c2, c3, c4)
    assert [r["case_id"] for r in body["rows"]] == ["c2", "c3", "c4"]


def test_rag_metrics_endpoint_handles_missing_file(app_client):
    client, _, _ = app_client
    body = client.get("/dashboard/metrics/rag").json()
    assert body == {"count": 0, "rows": [], "source": "rag_final_metrics.csv"}


# ---------------------------------------------------------------------------
# Rate-limit middleware
# ---------------------------------------------------------------------------
def test_middleware_adds_rate_headers_on_success(app_client):
    client, _, _ = app_client
    r = client.get("/ping")
    assert r.status_code == 200
    assert "X-RateLimit-Remaining" in r.headers
    assert r.headers["X-RateLimit-Tier"] == "default"


def test_middleware_bypasses_health():
    """/health must never 429 — it's the probe path."""
    from utils.rate_limiter import RateLimiter
    # Burst=1 limiter so we'd normally reject on the second call.
    tight = RateLimiter({"rate_limit": {
        "default": {"requests_per_minute": 1, "burst": 1, "burst_multiplier": 1.0},
    }})
    app = _make_app(limiter=tight)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    client = TestClient(app)
    for _ in range(3):
        r = client.get("/health")
        assert r.status_code == 200


def test_middleware_429_payload_and_headers():
    from utils.rate_limiter import RateLimiter
    tight = RateLimiter({"rate_limit": {
        "default": {"requests_per_minute": 1, "burst": 1, "burst_multiplier": 1.0},
    }})
    app = _make_app(limiter=tight)
    client = TestClient(app)

    r1 = client.get("/ping")
    assert r1.status_code == 200
    r2 = client.get("/ping")
    assert r2.status_code == 429
    body = r2.json()
    assert body["detail"] == "rate_limit_exceeded"
    assert body["tier"] == "default"
    assert float(body["retry_after_sec"]) > 0
    assert "Retry-After" in r2.headers


def test_middleware_admin_tier_for_dashboard_path(app_client):
    """/dashboard/* → admin bucket (separate from /ping's default bucket)."""
    client, _, _ = app_client
    r = client.get("/dashboard/alert-rules")
    assert r.status_code == 200
    assert r.headers.get("X-RateLimit-Tier") == "admin"


def test_middleware_uses_user_id_header():
    from utils.rate_limiter import RateLimiter
    tight = RateLimiter({"rate_limit": {
        "default": {"requests_per_minute": 1, "burst": 1, "burst_multiplier": 1.0},
    }})
    app = _make_app(limiter=tight)
    client = TestClient(app)

    # Same bucket per user — second call from 'alice' denied.
    assert client.get("/ping", headers={"X-User-Id": "alice"}).status_code == 200
    assert client.get("/ping", headers={"X-User-Id": "alice"}).status_code == 429
    # 'bob' gets a fresh bucket.
    assert client.get("/ping", headers={"X-User-Id": "bob"}).status_code == 200
