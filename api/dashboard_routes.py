"""api/dashboard_routes.py
============================
Read-only endpoints the dashboard consumes.

Everything here reads from on-disk artefacts (telemetry.jsonl, CSV metric
logs, the circuit-breaker registry) — no mutation, no pipeline execution.
That keeps the surface easy to reason about and lets the dashboard poll at
modest frequency without worrying about side effects.

Routes
------
GET /dashboard/summary        — single-shot snapshot for the home page
GET /dashboard/alerts         — list of fired Alerts from the current snapshot
GET /dashboard/recent-runs    — paginated fusion decisions (live feed)
GET /dashboard/circuit-breakers — state + counters per registered breaker
GET /dashboard/rate-limits    — current token-bucket levels (admin page)
GET /dashboard/metrics/output-guard — last N rows of output_security_metrics.csv
GET /dashboard/metrics/rag    — last N rows of rag_final_metrics.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from monitoring.alert_rules import build_snapshot_from_events, evaluate, load_rules
from schemas import telemetry_schema as ts
from utils.fallback_handler import _REGISTRY as _BREAKER_REGISTRY  # noqa: SLF001
from utils.rate_limiter import default_limiter


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUNS_DIR = _PROJECT_ROOT / "runs"

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_events(run_id: Optional[str], limit: int) -> List[Dict[str, Any]]:
    """Read raw telemetry events. Returns newest-first, capped to `limit`."""
    try:
        events = ts.read_events(run_id=run_id)
    except FileNotFoundError:
        return []
    # read_events yields in append order; newest-first is more useful for UIs.
    return list(reversed(events))[:limit]


def _tail_csv(path: Path, limit: int) -> List[Dict[str, str]]:
    """Read the last `limit` rows of a CSV (header preserved)."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-limit:]


# ---------------------------------------------------------------------------
# Summary — the dashboard home page calls this every few seconds.
# ---------------------------------------------------------------------------
@router.get("/summary")
def get_summary(
    run_id: Optional[str] = Query(None, description="Scope metrics to a single run_id."),
    event_limit: int = Query(500, ge=1, le=10_000,
                             description="Max telemetry events inspected."),
) -> Dict[str, Any]:
    """Aggregated snapshot: counts, rates, fired alerts, breaker states."""
    events = _read_events(run_id=run_id, limit=event_limit)
    snapshot = build_snapshot_from_events(events)
    fired = [a.to_dict() for a in evaluate(snapshot)]

    breakers = [{
        "name": name,
        "state": br.stats().state.value,
        "consecutive_failures": br.stats().consecutive_failures,
        "total_short_circuits": br.stats().total_short_circuits,
    } for name, br in _BREAKER_REGISTRY.items()]

    # Derive count fields the dashboard pages display as absolute numbers.
    total = int(snapshot.get("total_requests", 0))
    snapshot["block_count"] = int(round(snapshot.get("block_rate", 0.0) * total))
    snapshot["sanitize_count"] = int(round(snapshot.get("sanitize_rate", 0.0) * total))
    snapshot["allow_count"] = int(round(snapshot.get("allow_rate", 0.0) * total))
    # Alias — both pages use avg_total_latency_ms; snapshot builder stores avg_latency_ms.
    snapshot["avg_total_latency_ms"] = snapshot.get("avg_latency_ms", 0.0)

    return {
        "run_id": run_id,
        "event_window": len(events),
        "metrics": snapshot,      # structured access for API consumers
        "alerts": fired,
        "breakers": breakers,
        **snapshot,               # flat access for dashboard pages (1_home, 4_live_monitor)
    }


# ---------------------------------------------------------------------------
# Alerts — same data as /summary but trimmed; useful for a toast/badge.
# ---------------------------------------------------------------------------
@router.get("/alerts")
def get_alerts(
    run_id: Optional[str] = Query(None),
    min_severity: str = Query("info", pattern="^(info|warn|critical)$"),
    event_limit: int = Query(500, ge=1, le=10_000),
) -> Dict[str, Any]:
    order = {"info": 0, "warn": 1, "critical": 2}
    threshold = order.get(min_severity, 0)
    events = _read_events(run_id=run_id, limit=event_limit)
    snapshot = build_snapshot_from_events(events)
    alerts = [a.to_dict() for a in evaluate(snapshot)]
    alerts = [a for a in alerts if order.get(a["severity"], 0) >= threshold]
    return {"count": len(alerts), "alerts": alerts}


@router.get("/alert-rules")
def list_alert_rules() -> Dict[str, Any]:
    """Expose the currently loaded rules so the UI can show 'what triggers when'."""
    rules = load_rules()
    return {"count": len(rules), "rules": rules}


# ---------------------------------------------------------------------------
# Recent runs — newest fusion decisions, paginated.
# ---------------------------------------------------------------------------
@router.get("/recent-runs")
def get_recent_runs(
    run_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    event_limit: int = Query(2000, ge=1, le=50_000),
) -> Dict[str, Any]:
    events = _read_events(run_id=run_id, limit=event_limit)
    decisions = [e for e in events if e.get("kind") == "fusion_decision"][:limit]
    # Trim evidence payloads — the live feed doesn't need 4KB per row.
    for d in decisions:
        evidence = d.get("evidence") or []
        if isinstance(evidence, list) and len(evidence) > 3:
            d["evidence"] = evidence[:3]
    return {"count": len(decisions), "decisions": decisions}


# ---------------------------------------------------------------------------
# Circuit breakers + rate limiter — operational telemetry for admin page.
# ---------------------------------------------------------------------------
@router.get("/circuit-breakers")
def get_circuit_breakers() -> Dict[str, Any]:
    out: List[Dict[str, Any]] = []
    for name, br in _BREAKER_REGISTRY.items():
        s = br.stats()
        out.append({
            "name": name,
            "state": s.state.value,
            "consecutive_failures": s.consecutive_failures,
            "consecutive_successes": s.consecutive_successes,
            "total_calls": s.total_calls,
            "total_failures": s.total_failures,
            "total_short_circuits": s.total_short_circuits,
            "failure_threshold": br.failure_threshold,
            "open_cooldown_seconds": br.open_cooldown_seconds,
        })
    return {"count": len(out), "breakers": out}


@router.get("/rate-limits")
def get_rate_limits() -> Dict[str, Any]:
    return {"buckets": default_limiter().snapshot()}


# ---------------------------------------------------------------------------
# Metric tailers — generic CSV peek for the module detail pages.
# ---------------------------------------------------------------------------
@router.get("/metrics/output-guard")
def get_output_guard_metrics(limit: int = Query(100, ge=1, le=5000)) -> Dict[str, Any]:
    path = _RUNS_DIR / "output_security_metrics.csv"
    rows = _tail_csv(path, limit)
    return {"count": len(rows), "rows": rows, "source": path.name}


@router.get("/metrics/output-guard/explain")
def get_output_guard_explain(limit: int = Query(100, ge=1, le=5000)) -> Dict[str, Any]:
    path = _RUNS_DIR / "output_explainability_log.csv"
    rows = _tail_csv(path, limit)
    return {"count": len(rows), "rows": rows, "source": path.name}


@router.get("/metrics/rag")
def get_rag_metrics(limit: int = Query(100, ge=1, le=5000)) -> Dict[str, Any]:
    path = _RUNS_DIR / "rag_final_metrics.csv"
    rows = _tail_csv(path, limit)
    return {"count": len(rows), "rows": rows, "source": path.name}


@router.get("/metrics/rag/explain")
def get_rag_explain(limit: int = Query(100, ge=1, le=5000)) -> Dict[str, Any]:
    path = _RUNS_DIR / "rag_explainability_log.csv"
    rows = _tail_csv(path, limit)
    return {"count": len(rows), "rows": rows, "source": path.name}


@router.get("/metrics/external-eval")
def get_external_eval_metrics(limit: int = Query(100, ge=1, le=5000)) -> Dict[str, Any]:
    path = _RUNS_DIR / "external_eval_results.csv"
    rows = _tail_csv(path, limit)
    return {"count": len(rows), "rows": rows, "source": path.name}


__all__ = ["router"]
