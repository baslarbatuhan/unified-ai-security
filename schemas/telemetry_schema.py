"""schemas/telemetry_schema.py
===============================
Central telemetry event schema + jsonl emitter.

Design goals
------------
1. **One format, many producers.** External eval runner, fusion gateway,
   module callbacks and the dashboard all speak the same shape so the
   reporting layer (Phase 4) can aggregate without case-splits.
2. **Append-only + daily rotation.** Dashboard tails the file; reporting
   reads historical partitions. Rotation keeps a single file manageable
   even under long eval runs (~10k events/day budget).
3. **PII-safe by default.** When `utils.log_sanitizer` is available,
   every event passes through `sanitize_event` before disk write. The
   import is lazy so this schema module has no hard dependency on the
   sanitizer implementation (Phase 0.4).
4. **Thread-safe append.** Writer uses a module-level lock; safe under
   FastAPI async workers and pytest parallel runs.

Four event kinds (discriminator: `kind`):
  - `request`         — incoming gateway request (prompt + target)
  - `module_result`   — one module's verdict on one request
  - `fusion_decision` — final decision after weighted fusion
  - `error`           — any uncaught exception inside a module or the gateway

All share a `run_id` that groups events belonging to the same evaluation run.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
TELEMETRY_DIR = _PROJECT_ROOT / "logs"
TELEMETRY_FILE = TELEMETRY_DIR / "system_telemetry.jsonl"


# ---------------------------------------------------------------------------
# Event models
# ---------------------------------------------------------------------------
EventKind = Literal["request", "module_result", "fusion_decision", "error"]
Decision = Literal["allow", "sanitize", "block", "flag"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _new_event_id() -> str:
    return uuid.uuid4().hex[:16]


class BaseEvent(BaseModel):
    """Fields shared by every telemetry event."""

    event_id: str = Field(default_factory=_new_event_id)
    run_id: str = Field(..., description="Groups events in a single evaluation run.")
    timestamp: str = Field(default_factory=_now_iso)
    kind: EventKind
    target_id: Optional[str] = Field(
        default=None,
        description="Chatbot identifier from external_eval/targets.yaml. None for internal tests.",
    )
    attack_id: Optional[str] = Field(
        default=None,
        description="Attack suite sample id when the event belongs to a scripted probe.",
    )


class RequestEvent(BaseEvent):
    kind: Literal["request"] = "request"
    prompt: str
    prompt_char_count: int = Field(
        ..., ge=0, description="Populate at construction; serves as a sanity check."
    )
    has_retrieved_docs: bool = False
    retrieved_doc_count: int = 0
    session_role: str = "basic"


class ModuleResultEvent(BaseEvent):
    kind: Literal["module_result"] = "module_result"
    module: str  # prompt_guard | rag_guard | output_agency | output_guard | ...
    risk_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    decision: Decision
    latency_ms: int = Field(..., ge=0)
    evidence: List[str] = Field(default_factory=list)
    # Free-form module payload for explainability — keep small (<4 KB recommended)
    details: Dict[str, Any] = Field(default_factory=dict)


class FusionDecisionEvent(BaseEvent):
    kind: Literal["fusion_decision"] = "fusion_decision"
    fused_risk_score: float = Field(..., ge=0.0, le=1.0)
    decision: Decision
    prompt_score: float = 0.0
    rag_score: float = 0.0
    agency_score: float = 0.0
    output_score: float = 0.0
    evidence: List[str] = Field(default_factory=list)
    latency_ms_total: int = Field(..., ge=0)


class ErrorEvent(BaseEvent):
    kind: Literal["error"] = "error"
    where: str = Field(..., description="Module or stage that raised.")
    error_type: str
    message: str
    # No traceback by default — set via details if caller wants it, mindful of PII.
    details: Dict[str, Any] = Field(default_factory=dict)


# Union covering every kind a producer may emit.
TelemetryEvent = Union[RequestEvent, ModuleResultEvent, FusionDecisionEvent, ErrorEvent]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
_WRITE_LOCK = threading.Lock()


def _sanitize(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort PII scrub. Delegates to utils.log_sanitizer if available.

    The lazy import keeps this schema module importable even before Phase 0.4
    lands.  When the sanitizer is missing, the event is written unchanged — the
    schema is the stable contract, the sanitizer is an auditability add-on.
    """
    try:
        from utils.log_sanitizer import sanitize_event  # type: ignore
    except Exception:
        return payload
    try:
        return sanitize_event(payload)
    except Exception:
        # Never let sanitization bugs block telemetry.
        return payload


def _rotated_path(when: Optional[datetime] = None) -> Path:
    """Daily rotation — telemetry lives in `system_telemetry.jsonl` for "today"
    and gets stamped with a date suffix on other days.

    We always write to the un-dated symlink-ish path (`system_telemetry.jsonl`)
    for dashboards to tail, then periodically move it to a dated file. The move
    is done by an external cron / dashboard tick, not inline, to keep the write
    path lock-free.

    For now we append to a single file and expose `get_daily_path` for callers
    that want to archive explicitly.
    """
    return TELEMETRY_FILE


def get_daily_path(when: Optional[datetime] = None) -> Path:
    """Dated archive path used by rotation helpers."""
    when = when or datetime.now(timezone.utc)
    return TELEMETRY_DIR / f"system_telemetry.{when:%Y-%m-%d}.jsonl"


def emit_telemetry(event: TelemetryEvent) -> None:
    """Append a telemetry event to the jsonl log.

    Guarantees:
      * atomic per-line append (held by `_WRITE_LOCK`)
      * never raises — telemetry must not break the request path
      * sanitized when `utils.log_sanitizer` is installed
    """
    try:
        payload = event.model_dump(mode="json")
        payload = _sanitize(payload)
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except Exception as exc:  # pragma: no cover — defensive
        # Fall back to a minimal error stub so we at least see *something*.
        line = json.dumps(
            {
                "kind": "error",
                "timestamp": _now_iso(),
                "where": "emit_telemetry",
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
            ensure_ascii=False,
        )

    TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
    path = _rotated_path()
    with _WRITE_LOCK:
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:  # pragma: no cover
            # Last-ditch: write to stderr so the operator sees the failure.
            print(f"[telemetry] write failed: {exc}", file=sys.stderr)


def read_events(
    path: Optional[Path] = None,
    *,
    run_id: Optional[str] = None,
    kinds: Optional[List[EventKind]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Convenience reader used by reporting / dashboard.

    Returns dicts (not Pydantic objects) to keep downstream code loose; the
    schema is enforced on *write*, not on read.  Unknown fields are tolerated
    because the schema may grow over time.
    """
    path = path or _rotated_path()
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if run_id is not None and ev.get("run_id") != run_id:
                continue
            if kinds is not None and ev.get("kind") not in kinds:
                continue
            out.append(ev)
            if limit is not None and len(out) >= limit:
                break
    return out


# ---------------------------------------------------------------------------
# run_id helpers
# ---------------------------------------------------------------------------
def new_run_id(prefix: str = "run") -> str:
    """Build a sortable run id: `<prefix>_YYYYmmddHHMMSS_<hex6>`."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:6]}"


__all__ = [
    "BaseEvent",
    "RequestEvent",
    "ModuleResultEvent",
    "FusionDecisionEvent",
    "ErrorEvent",
    "TelemetryEvent",
    "EventKind",
    "Decision",
    "emit_telemetry",
    "read_events",
    "new_run_id",
    "get_daily_path",
    "TELEMETRY_FILE",
    "TELEMETRY_DIR",
]
