"""tests/test_phase5_routes.py
Smoke + behaviour tests for the three new routers:
    api/routes_runs.py
    api/routes_reports.py
    api/routes_targets.py

We mount each router on a bare FastAPI app and use TestClient. Telemetry
and reports directories are redirected to tmp_path so tests never touch
the real on-disk state.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Telemetry fixture — patch the schema's TELEMETRY_FILE before importing routers
# ---------------------------------------------------------------------------
@pytest.fixture
def telemetry_file(tmp_path, monkeypatch):
    """Redirect ts.read_events() to a tmp jsonl, return its path."""
    p = tmp_path / "system_telemetry.jsonl"
    import schemas.telemetry_schema as ts_mod
    monkeypatch.setattr(ts_mod, "TELEMETRY_FILE", p)
    return p


def _write_events(path: Path, events):
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


# ---------------------------------------------------------------------------
# /runs
# ---------------------------------------------------------------------------
def _make_run(run_id: str, decision="allow", fused=0.1, ts0="2026-04-25T10:00:00Z"):
    return [
        {"run_id": run_id, "kind": "request", "timestamp": ts0,
         "prompt": "hi", "prompt_char_count": 2,
         "has_retrieved_docs": False, "retrieved_doc_count": 0, "session_role": "basic"},
        {"run_id": run_id, "kind": "module_result", "timestamp": ts0,
         "module": "prompt_guard", "risk_score": fused, "confidence": 0.9,
         "decision": decision, "latency_ms": 50, "evidence": []},
        {"run_id": run_id, "kind": "fusion_decision", "timestamp": ts0,
         "fused_risk_score": fused, "decision": decision,
         "prompt_score": fused, "rag_score": 0, "agency_score": 0, "output_score": 0,
         "evidence": [], "latency_ms_total": 60},
    ]


@pytest.fixture
def runs_client(telemetry_file):
    from api.routes_runs import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app), telemetry_file


def test_list_runs_empty_file_returns_zero(runs_client):
    client, _ = runs_client
    r = client.get("/runs")
    assert r.status_code == 200
    body = r.json()
    assert body == {"total": 0, "limit": 50, "offset": 0, "events_scanned": 0, "runs": []}


def test_list_runs_groups_events_by_run_id(runs_client):
    client, tf = runs_client
    events = (
        _make_run("run_A", decision="allow", fused=0.10, ts0="2026-04-25T10:00:00Z")
        + _make_run("run_B", decision="block", fused=0.95, ts0="2026-04-25T11:00:00Z")
    )
    _write_events(tf, events)

    r = client.get("/runs")
    body = r.json()
    assert body["total"] == 2
    # Newest first
    assert body["runs"][0]["run_id"] == "run_B"
    assert body["runs"][0]["decision"] == "block"
    assert body["runs"][0]["fused_risk_score"] == pytest.approx(0.95)
    assert body["runs"][1]["run_id"] == "run_A"


def test_list_runs_filters_by_decision(runs_client):
    client, tf = runs_client
    _write_events(tf,
                  _make_run("run_A", decision="allow") + _make_run("run_B", decision="block"))
    r = client.get("/runs?decision=block")
    body = r.json()
    assert body["total"] == 1
    assert body["runs"][0]["run_id"] == "run_B"


def test_get_run_returns_full_timeline(runs_client):
    client, tf = runs_client
    _write_events(tf, _make_run("run_X", decision="flag", fused=0.65))
    r = client.get("/runs/run_X")
    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == "run_X"
    assert body["event_count"] == 3
    kinds = {ev["kind"] for ev in body["events"]}
    assert kinds == {"request", "module_result", "fusion_decision"}


def test_get_run_404_for_unknown(runs_client):
    client, _ = runs_client
    r = client.get("/runs/nonexistent")
    assert r.status_code == 404


def test_get_run_summary_404_for_unknown(runs_client):
    client, _ = runs_client
    r = client.get("/runs/nonexistent/summary")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /reports
# ---------------------------------------------------------------------------
@pytest.fixture
def reports_client(tmp_path, monkeypatch):
    """Redirect _REPORTS_DIR to a tmp dir we control."""
    import api.routes_reports as rr
    rdir = tmp_path / "reports"
    rdir.mkdir()
    monkeypatch.setattr(rr, "_REPORTS_DIR", rdir)
    app = FastAPI()
    app.include_router(rr.router)
    return TestClient(app), rdir


def test_list_reports_empty_dir(reports_client):
    client, _ = reports_client
    body = client.get("/reports").json()
    assert body == {"reports": [], "total": 0}


def test_list_reports_extracts_first_heading_as_title(reports_client):
    client, rdir = reports_client
    (rdir / "alpha.md").write_text("# Alpha Beta\n\nbody", encoding="utf-8")
    (rdir / "no_heading.md").write_text("body only", encoding="utf-8")
    body = client.get("/reports").json()
    titles = {r["name"]: r["title"] for r in body["reports"]}
    assert titles["alpha.md"] == "Alpha Beta"
    # Falls back to filename-derived title
    assert titles["no_heading.md"] == "No Heading"


def test_get_report_returns_markdown(reports_client):
    client, rdir = reports_client
    (rdir / "x.md").write_text("# Hi\n\nworld", encoding="utf-8")
    r = client.get("/reports/x.md")
    assert r.status_code == 200
    assert "text/markdown" in r.headers["content-type"]
    assert r.text.startswith("# Hi")


def test_get_report_rejects_path_traversal(reports_client):
    client, _ = reports_client
    for evil in ("../etc/passwd", "..\\windows", "x/y.md", "/abs.md", "..%2Fy.md"):
        r = client.get(f"/reports/{evil}")
        assert r.status_code in (400, 404), f"{evil} should be blocked, got {r.status_code}"


def test_get_report_rejects_non_md(reports_client):
    client, rdir = reports_client
    (rdir / "x.txt").write_text("body", encoding="utf-8")
    r = client.get("/reports/x.txt")
    assert r.status_code == 400


def test_get_report_404_for_missing(reports_client):
    client, _ = reports_client
    r = client.get("/reports/nope.md")
    assert r.status_code == 404


def test_download_report_sets_attachment_header(reports_client):
    client, rdir = reports_client
    (rdir / "y.md").write_text("# Y\n\n", encoding="utf-8")
    r = client.get("/reports/y.md/download")
    assert r.status_code == 200
    assert 'attachment' in r.headers["content-disposition"]
    assert 'filename="y.md"' in r.headers["content-disposition"]


# ---------------------------------------------------------------------------
# /targets
# ---------------------------------------------------------------------------
@pytest.fixture
def targets_client(tmp_path, monkeypatch):
    """Point the loader at a tmp targets.yaml."""
    import external_eval.target_loader as tl
    p = tmp_path / "targets.yaml"
    p.write_text(
        "version: 1\n"
        "targets:\n"
        "  - id: mock1\n"
        "    name: Mock One\n"
        "    type: mock\n"
        "    enabled: true\n"
        "    timeout_seconds: 5.0\n"
        "  - id: api1\n"
        "    name: API One\n"
        "    type: api\n"
        "    enabled: false\n"
        "    endpoint: https://example.test/chat\n"
        "    timeout_seconds: 10.0\n"
        "    auth:\n"
        "      type: bearer\n"
        "      token: super-secret-must-be-redacted\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(tl, "DEFAULT_TARGETS_PATH", p)

    from api.routes_targets import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app), p


def test_list_targets_returns_both(targets_client):
    client, _ = targets_client
    body = client.get("/targets").json()
    assert body["total"] == 2
    ids = {t["id"] for t in body["targets"]}
    assert ids == {"mock1", "api1"}


def test_list_targets_enabled_only(targets_client):
    client, _ = targets_client
    body = client.get("/targets?enabled_only=true").json()
    assert body["total"] == 1
    assert body["targets"][0]["id"] == "mock1"


def test_get_target_redacts_token(targets_client):
    client, _ = targets_client
    body = client.get("/targets/api1").json()
    assert body["id"] == "api1"
    assert body["auth"]["token"] == "***redacted***"
    # Type field still visible
    assert body["auth"]["type"] == "bearer"


def test_get_target_404(targets_client):
    client, _ = targets_client
    r = client.get("/targets/does-not-exist")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /targets POST + DELETE — CRUD round-trip
# ---------------------------------------------------------------------------
def test_upsert_target_inserts_then_appears_in_list(targets_client):
    client, path = targets_client
    payload = {
        "id": "added_via_post",
        "name": "Added via POST",
        "type": "mock",
        "enabled": True,
        "timeout_seconds": 7.0,
    }
    r = client.post("/targets", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["id"] == "added_via_post"

    listing = client.get("/targets").json()
    ids = {t["id"] for t in listing["targets"]}
    assert "added_via_post" in ids


def test_upsert_target_validation_error_returns_422(targets_client):
    client, _ = targets_client
    # Missing required `id` field.
    r = client.post("/targets", json={"type": "mock"})
    assert r.status_code == 422


def test_delete_target_removes_it(targets_client):
    client, _ = targets_client
    r = client.delete("/targets/mock1")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert client.get("/targets/mock1").status_code == 404


def test_delete_target_404_for_unknown(targets_client):
    client, _ = targets_client
    r = client.delete("/targets/does-not-exist")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /runs/start + /runs/{id}/status — background launch
# ---------------------------------------------------------------------------
def test_start_run_writes_status_and_returns_run_id(tmp_path, monkeypatch):
    """`POST /runs/start` should write status.json synchronously and spawn the
    runner asynchronously. We stub `_supervise` so no real subprocess fires."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api import routes_runs as rr

    monkeypatch.setattr(rr, "_RUNS_DIR", tmp_path)
    # Replace the supervisor with a no-op so the test stays hermetic.
    monkeypatch.setattr(rr, "_supervise", lambda *a, **kw: None)

    app = FastAPI()
    app.include_router(rr.router)
    client = TestClient(app)

    r = client.post(
        "/runs/start",
        json={"target": "mock_echo", "suite": "prompt_injection", "max_attacks": 2},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["state"] == "queued"
    rid = body["run_id"]
    assert rid.startswith("ext_mock_echo_")

    status = client.get(f"/runs/{rid}/status").json()
    assert status["state"] == "queued"
    assert status["target"] == "mock_echo"


def test_status_404_for_unknown_run(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api import routes_runs as rr

    monkeypatch.setattr(rr, "_RUNS_DIR", tmp_path)
    app = FastAPI()
    app.include_router(rr.router)
    client = TestClient(app)
    r = client.get("/runs/no-such-run/status")
    assert r.status_code == 404
