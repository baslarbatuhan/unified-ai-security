"""api/routes_decisions.py
============================
Read-only access to per-decision audit traces.

The gateway's `FusionEngine.analyze()` writes one row per /analyze call
to `runs/<run_id>/decision_trace.csv` (when `run_id` is non-"live"). This
router exposes that data to the dashboard's drill-down panels:

    GET /decisions/{run_id}                — full trace for a run (paginated)
    GET /decisions/{run_id}/{case_id}/trace — single case's trace

Both endpoints are pure-read; no side-effects, no auth required beyond
whatever the gateway-level middleware applies. The JSON shape matches
`utils.run_manifest.DECISION_TRACE_FIELDS` plus a parsed `module_risks`
list (the CSV stores it as a JSON-encoded string for portability).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from utils.run_manifest import (
    DECISION_TRACE_FIELDS,
    read_decision_trace,
)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUNS_DIR = _PROJECT_ROOT / "runs"


router = APIRouter(prefix="/decisions", tags=["decisions"])


def _run_dir(run_id: str) -> Path:
    """Resolve `runs/<run_id>/`. Surfaces a 404 if the run dir doesn't
    exist — saves the caller from peeking at the CSV path directly."""
    rd = _RUNS_DIR / run_id
    if not rd.exists() or not rd.is_dir():
        raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id!r}")
    return rd


@router.get("/{run_id}")
def list_run_decisions(
    run_id: str,
    limit: int = Query(200, ge=1, le=2000),
    case_id: Optional[str] = Query(
        None, description="Optional filter to a single case_id."
    ),
) -> Dict[str, Any]:
    """Return all decision-trace rows for a run. Newest-last in file
    order; the dashboard reverses for display when needed.

    Returns `{"count": N, "rows": [...]}`. Each row is a dict with the
    columns in `DECISION_TRACE_FIELDS` plus a parsed `module_risks` list.
    """
    rd = _run_dir(run_id)
    rows = read_decision_trace(rd, case_id=case_id)
    rows = rows[-limit:] if limit else rows
    # Normalise field name for downstream consumers: the CSV column is
    # `module_risks_json` (so spreadsheets can see the JSON text); the
    # API surface drops the suffix and ships the parsed structure.
    for r in rows:
        if "module_risks_json" in r and "module_risks" not in r:
            r["module_risks"] = r.pop("module_risks_json")
    return {
        "run_id": run_id,
        "count": len(rows),
        "rows": rows,
        "schema_fields": DECISION_TRACE_FIELDS,
    }


@router.get("/{run_id}/{case_id}/trace")
def get_case_trace(run_id: str, case_id: str) -> Dict[str, Any]:
    """Single-case decision trace.

    Returns the latest matching row (a case_id may appear more than once
    if the run was retried — the dashboard's drill-down wants the most
    recent attempt). 404 when no trace exists for that case in this run.
    """
    rd = _run_dir(run_id)
    rows = read_decision_trace(rd, case_id=case_id)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"no decision trace for case {case_id!r} in run {run_id!r}",
        )
    last = rows[-1]
    if "module_risks_json" in last and "module_risks" not in last:
        last["module_risks"] = last.pop("module_risks_json")
    return {"run_id": run_id, "case_id": case_id, "trace": last}
