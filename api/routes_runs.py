"""api/routes_runs.py
Run-history + run-launch endpoints.

The /runs surface is run-centric: one row per run, linkable, drill-down
to the underlying telemetry events. It also fronts the external_eval
runner so the dashboard can launch a suite without spawning a CLI.

Routes
------
GET    /runs                       — paginated list of runs (newest first)
GET    /runs/{run_id}              — every event for the run, in order
GET    /runs/{run_id}/summary      — one-line summary used as the list row
GET    /runs/{run_id}/status       — live status of a launched run (queued/
                                     running/done/failed)
POST   /runs/start                 — launch an external_eval suite as a
                                     background subprocess; returns run_id
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from schemas import telemetry_schema as ts


router = APIRouter(prefix="/runs", tags=["runs"])

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUNS_DIR = _PROJECT_ROOT / "runs"


# ---------------------------------------------------------------------------
# Internal — group raw events into per-run summaries
# ---------------------------------------------------------------------------
def _group_runs(events: List[Dict[str, Any]]) -> "OrderedDict[str, Dict[str, Any]]":
    """Bucket events by run_id, derive a small summary per run."""
    runs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for ev in events:
        rid = ev.get("run_id")
        if not rid:
            continue
        bucket = runs.setdefault(rid, {
            "run_id": rid,
            "started_at": ev.get("timestamp"),
            "ended_at": ev.get("timestamp"),
            "event_count": 0,
            "decision": None,
            "fused_risk_score": None,
            "prompt_score": 0.0,
            "rag_score": 0.0,
            "agency_score": 0.0,
            "output_score": 0.0,
            "latency_ms": None,
            "modules_seen": [],
            "target_id": ev.get("target_id"),
            "attack_id": ev.get("attack_id"),
        })
        bucket["event_count"] += 1
        ts_ = ev.get("timestamp")
        if ts_ and (bucket["started_at"] is None or ts_ < bucket["started_at"]):
            bucket["started_at"] = ts_
        if ts_ and (bucket["ended_at"] is None or ts_ > bucket["ended_at"]):
            bucket["ended_at"] = ts_

        kind = ev.get("kind")
        if kind == "module_result":
            mod = ev.get("module")
            if mod and mod not in bucket["modules_seen"]:
                bucket["modules_seen"].append(mod)
        elif kind == "fusion_decision":
            # Fusion is the "verdict" event — it shadows any earlier value.
            bucket["decision"] = ev.get("decision")
            bucket["fused_risk_score"] = ev.get("fused_risk_score")
            bucket["prompt_score"] = ev.get("prompt_score", 0.0)
            bucket["rag_score"] = ev.get("rag_score", 0.0)
            bucket["agency_score"] = ev.get("agency_score", 0.0)
            bucket["output_score"] = ev.get("output_score", 0.0)
            bucket["latency_ms"] = ev.get("latency_ms_total")
    return runs


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.get("")
def list_runs(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    decision: Optional[str] = Query(None, description="Filter by final decision."),
    target_id: Optional[str] = Query(None, description="Filter by external target id."),
    event_scan_limit: int = Query(5000, ge=1, le=50_000,
                                  description="Telemetry events read; bigger = deeper history."),
) -> Dict[str, Any]:
    """List runs newest-first."""
    try:
        events = ts.read_events(limit=event_scan_limit)
    except FileNotFoundError:
        events = []

    runs_by_id = _group_runs(events)
    rows = list(runs_by_id.values())
    # Newest-first by started_at (ISO timestamps sort lexicographically).
    rows.sort(key=lambda r: r.get("started_at") or "", reverse=True)

    if decision:
        rows = [r for r in rows if (r.get("decision") or "") == decision]
    if target_id:
        rows = [r for r in rows if (r.get("target_id") or "") == target_id]

    total = len(rows)
    page = rows[offset: offset + limit]
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "events_scanned": len(events),
        "runs": page,
    }


@router.get("/{run_id}/summary")
def get_run_summary(run_id: str) -> Dict[str, Any]:
    """One-row summary for a single run."""
    try:
        events = ts.read_events(run_id=run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="telemetry log missing")
    if not events:
        raise HTTPException(status_code=404, detail=f"no events for run_id={run_id}")
    runs = _group_runs(events)
    summary = runs.get(run_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"no events for run_id={run_id}")
    return summary


@router.get("/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    """Full event timeline for a single run."""
    try:
        events = ts.read_events(run_id=run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="telemetry log missing")
    if not events:
        raise HTTPException(status_code=404, detail=f"no events for run_id={run_id}")
    return {
        "run_id": run_id,
        "event_count": len(events),
        "events": events,
    }


# ---------------------------------------------------------------------------
# Run launch — spawn external_eval/run_external_eval.py in the background
# ---------------------------------------------------------------------------
class StartRunRequest(BaseModel):
    target: str = Field(..., description="Target id from external_eval/targets.yaml")
    suite: str = Field("prompt_injection", description="Attack suite name.")
    max_attacks: int = Field(0, ge=0, description="0 = all cases in the suite.")
    target_has_tools: bool = Field(False, description="Override the target_has_tools gate.")
    run_id: Optional[str] = Field(None, description="Optional explicit run_id; auto-generated otherwise.")
    config_snapshot_path: Optional[str] = Field(
        None,
        description=(
            "Optional path to a config snapshot YAML (typically "
            "runs/<run_id>/config_used.yaml). Forwarded to the runner via "
            "--config-yaml so dashboard slider values reach the gateway."
        ),
    )


def _status_path(run_id: str) -> Path:
    return _RUNS_DIR / run_id / "status.json"


def _write_status(run_id: str, **fields: Any) -> None:
    p = _status_path(run_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {}
    if p.exists():
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload.update(fields)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _supervise(run_id: str, args: List[str]) -> None:
    """Run the CLI as a subprocess; record start/exit in the run's status.json."""
    log_path = _RUNS_DIR / run_id / "runner.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _write_status(run_id, state="running", started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    try:
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.run(
                args, stdout=logf, stderr=subprocess.STDOUT, cwd=str(_PROJECT_ROOT), check=False
            )
        _write_status(
            run_id,
            state="done" if proc.returncode == 0 else "failed",
            exit_code=proc.returncode,
            ended_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    except Exception as exc:  # noqa: BLE001 — record any spawn failure
        _write_status(
            run_id,
            state="failed",
            error=str(exc),
            ended_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )


@router.post("/start")
def start_run(req: StartRunRequest) -> Dict[str, Any]:
    """Launch the external_eval runner for (target, suite) in the background."""
    run_id = req.run_id or f"ext_{req.target}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "external_eval" / "run_external_eval.py"),
        "--target", req.target,
        "--suite", req.suite,
        "--max-attacks", str(req.max_attacks),
        "--run-id", run_id,
    ]
    if req.target_has_tools:
        cmd.append("--target-has-tools")
    # Default to the snapshot the dashboard wrote; explicit path wins.
    snapshot = req.config_snapshot_path or str(_RUNS_DIR / run_id / "config_used.yaml")
    if Path(snapshot).exists():
        cmd.extend(["--config-yaml", snapshot])

    _write_status(run_id, state="queued", target=req.target, suite=req.suite, command=" ".join(cmd))
    threading.Thread(target=_supervise, args=(run_id, cmd), daemon=True).start()
    return {"run_id": run_id, "state": "queued", "command": cmd}


@router.get("/{run_id}/status")
def get_run_status(run_id: str) -> Dict[str, Any]:
    """Live status for a launched run: queued / running / done / failed."""
    p = _status_path(run_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"no status for run_id={run_id}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="status.json corrupted")
