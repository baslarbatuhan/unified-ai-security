"""utils/run_manifest.py
==========================
Per-run artefact manifest + flat registry index.

Each completed run writes:
    runs/<run_id>/manifest.json   — what this run produced + how to find it
    runs/<run_id>/results.csv     — rows scoped to THIS run (normalised)
    runs/_registry.jsonl          — append-only index of all known runs

The Results page in the dashboard reads the registry to populate its
run picker without scanning the whole runs/ tree, then loads the per-run
`results.csv` directly. Legacy aggregate CSVs (`runs/external_eval_results.csv`,
`runs/rag_final_metrics.csv`, …) keep flowing in parallel for backward
compatibility — nothing here removes them.

Schema version stays in `MANIFEST_SCHEMA_VERSION`. Bump it when the
manifest payload shape changes; the dashboard reads `schema_version`
and degrades gracefully on older entries.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
RESULTS_FILENAME = "results.csv"
REGISTRY_FILENAME = "_registry.jsonl"
DECISION_TRACE_FILENAME = "decision_trace.csv"

# Stable column order. The engine appends one row per /analyze call when
# the caller supplies a non-"live" run_id. Schema-drift rotation works the
# same way as `external_eval_results.csv` (caller does the rename); the
# fields below are the source of truth at write-time.
DECISION_TRACE_FIELDS = [
    "case_id",
    "target_id",
    "timestamp",
    "final_decision",      # 3-class: allow / sanitize / block
    "decision_band",       # 4-class: allow / sanitize / flag / block
    "fused_risk",
    "weighted_sum",        # pre-override fused
    "override_applied",    # "critical" | "elevated" | "none"
    "triggering_module",   # module with the highest risk_score
    "triggering_band",     # band that triggering_module's risk lands in
    "prompt_score",
    "rag_score",
    "agency_score",
    "output_score",
    "fusion_formula",      # human-readable formula snapshot
    "module_risks_json",   # compact JSON: [{module, risk_score, decision, top_evidence}]
    "latency_ms",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_results_csv(
    run_dir: Path,
    rows: List[Dict[str, Any]],
    fieldnames: List[str],
) -> Path:
    """Write the run-scoped results.csv into the run directory.

    Mirrors the legacy `runs/external_eval_results.csv` schema exactly, just
    filtered to this run's rows. Keeps downstream analysis code (pandas
    consumers, the Results page) compatible without a translation step.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / RESULTS_FILENAME
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return out_path


def write_manifest(
    run_dir: Path,
    *,
    run_id: str,
    target_id: Optional[str],
    suite: Optional[str],
    started_at: Optional[str],
    ended_at: Optional[str],
    exit_code: Optional[int],
    n_cases: int,
    n_rows: int,
    artefacts: Optional[Dict[str, Optional[str]]] = None,
    sources: Optional[Dict[str, str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write manifest.json describing this run's outputs and provenance.

    `artefacts` lists files inside `run_dir` that consumers can rely on;
    `sources` lists external CSVs the dashboard may cross-reference (for
    rag_final_metrics, output_security_metrics, etc.) — paths are stored
    relative to the project root so the manifest survives `runs/` moves.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect what's actually there if caller didn't pin the list.
    detected: Dict[str, Optional[str]] = {
        "results_csv": RESULTS_FILENAME if (run_dir / RESULTS_FILENAME).exists() else None,
        "config_used": "config_used.yaml" if (run_dir / "config_used.yaml").exists() else None,
        "runner_log": "runner.log" if (run_dir / "runner.log").exists() else None,
        "status_json": "status.json" if (run_dir / "status.json").exists() else None,
    }
    if artefacts:
        detected.update(artefacts)

    payload: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "target_id": target_id,
        "suite": suite,
        "started_at": started_at,
        "ended_at": ended_at or _utc_now_iso(),
        "exit_code": exit_code,
        "n_cases": int(n_cases or 0),
        "n_rows": int(n_rows or 0),
        "artefacts": detected,
        "sources": sources or {},
    }
    if extra:
        payload["extra"] = extra

    out_path = run_dir / MANIFEST_FILENAME
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def append_registry_entry(
    runs_dir: Path,
    *,
    run_id: str,
    target_id: Optional[str],
    suite: Optional[str],
    started_at: Optional[str],
    ended_at: Optional[str],
    exit_code: Optional[int],
    n_cases: int,
    n_rows: int,
    relative_path: str,
) -> Path:
    """Append a one-line summary to runs/_registry.jsonl.

    The registry is the dashboard's authoritative list of "what runs
    exist". Append-only is intentional: the file doubles as an audit
    log. Concurrent writers are rare (one runner per run) but safe —
    each line is one fsync, no partial JSON object can leak.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / REGISTRY_FILENAME
    entry = {
        "run_id": run_id,
        "target_id": target_id,
        "suite": suite,
        "started_at": started_at,
        "ended_at": ended_at or _utc_now_iso(),
        "exit_code": exit_code,
        "n_cases": int(n_cases or 0),
        "n_rows": int(n_rows or 0),
        "path": relative_path,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "registered_at": _utc_now_iso(),
    }
    line = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    with out_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return out_path


def read_registry(runs_dir: Path, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read registry entries, newest-first. Tolerant of corrupt lines.

    The dashboard calls this on every render of the run picker. Cheap:
    ~1KB per entry, file is append-only and grows linearly with runs.
    Tail-only access pattern matches typical UI usage (latest 50 runs).
    """
    path = runs_dir / REGISTRY_FILENAME
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Corrupted/truncated line — skip rather than fail the whole UI.
                continue
    out.reverse()  # newest-first
    if limit is not None:
        out = out[:limit]
    return out


def read_manifest(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Read a single run's manifest. Returns None when absent."""
    p = run_dir / MANIFEST_FILENAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def append_decision_trace(
    run_dir: Path,
    *,
    row: Dict[str, Any],
) -> Optional[Path]:
    """Append a single row to `runs/<run_id>/decision_trace.csv`.

    Best-effort: returns None on any I/O failure rather than raising —
    decision-trace logging must never break the gateway request path.
    Writes the header lazily when the file doesn't exist yet. The
    `row` dict is filtered to `DECISION_TRACE_FIELDS` so unknown keys
    don't drift into the on-disk schema.

    Schema drift: if the existing header differs from the current
    `DECISION_TRACE_FIELDS`, the legacy file is rotated to
    `decision_trace.csv.stale-<utc>` and a fresh file is started — same
    pattern used by `external_eval_results.csv`. Keeps newer columns
    additive without losing the old data.
    """
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / DECISION_TRACE_FILENAME

        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    first = f.readline().strip()
                existing = first.split(",") if first else []
                if existing != DECISION_TRACE_FIELDS:
                    stale = path.with_suffix(
                        path.suffix
                        + f".stale-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
                    )
                    path.rename(stale)
            except OSError:
                # Defensive — fall through, DictWriter will still work.
                pass

        write_header = not path.exists()
        # Restrict to known columns; coerce non-string values to str so
        # readers downstream (pandas, csv.DictReader) get consistent
        # types without nested quoting headaches.
        clean: Dict[str, str] = {}
        for k in DECISION_TRACE_FIELDS:
            v = row.get(k, "")
            if v is None:
                clean[k] = ""
            elif isinstance(v, (dict, list)):
                clean[k] = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
            else:
                clean[k] = str(v)

        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DECISION_TRACE_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(clean)
        return path
    except Exception:  # noqa: BLE001 — best-effort
        return None


def read_decision_trace(
    run_dir: Path, case_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Read `runs/<run_id>/decision_trace.csv`. Filters to `case_id` when
    provided. Tolerant of missing file / corrupt rows — returns []. The
    JSON column (`module_risks_json`) is parsed back to a Python list so
    callers don't deal with raw strings."""
    path = run_dir / DECISION_TRACE_FILENAME
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if case_id and (r.get("case_id") or "") != case_id:
                    continue
                # Parse the JSON column back.
                raw = r.get("module_risks_json") or ""
                if raw:
                    try:
                        r["module_risks_json"] = json.loads(raw)
                    except json.JSONDecodeError:
                        pass
                out.append(r)
    except OSError:
        return []
    return out


def filter_rows_by_run_id(
    csv_path: Path, run_id: str, *, run_id_col: str = "run_id"
) -> Iterable[Dict[str, str]]:
    """Stream rows from a multi-run aggregate CSV that match `run_id`.

    Used to materialise per-run results.csv from the legacy
    `runs/external_eval_results.csv` for runs that predate the manifest era.
    Generator-shaped so caller can collect or stream-write.
    """
    if not csv_path.exists():
        return
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get(run_id_col) or "") == run_id:
                yield row


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "RESULTS_FILENAME",
    "REGISTRY_FILENAME",
    "DECISION_TRACE_FILENAME",
    "DECISION_TRACE_FIELDS",
    "write_results_csv",
    "write_manifest",
    "append_registry_entry",
    "read_registry",
    "read_manifest",
    "append_decision_trace",
    "read_decision_trace",
    "filter_rows_by_run_id",
]
