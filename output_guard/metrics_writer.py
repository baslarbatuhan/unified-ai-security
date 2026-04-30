"""output_guard/metrics_writer.py
==================================
Append per-call rows to the two output-guard CSVs:

    runs/output_security_metrics.csv       — one row per analyze() call
    runs/output_explainability_log.csv     — one row per triggered flag

Kept in its own module so `output_analyzer.analyze` stays pure and easy to
test; the fusion engine and the external-eval runner both call
`record_result(...)` after the fact.
"""

from __future__ import annotations

import csv
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from output_guard.output_analyzer import OutputRiskResult

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUNS_DIR = _PROJECT_ROOT / "runs"

METRICS_PATH = _RUNS_DIR / "output_security_metrics.csv"
EXPLAIN_PATH = _RUNS_DIR / "output_explainability_log.csv"

_METRICS_FIELDS = [
    "run_id", "case_id", "target_id",
    "score", "decision", "output_chars", "latency_ms",
    "flag_pii", "flag_api_key", "flag_unsafe_instruction",
    "flag_downstream_injection", "flag_redirect_to_unknown",
    "evidence_top",
]
_EXPLAIN_FIELDS = [
    "run_id", "case_id", "target_id",
    "flag_name", "rule_or_subtype", "sample",
]

_LOCK = Lock()


def _ensure_writer(path: Path, fields: list[str]):
    """Append-mode writer with schema-drift protection.

    If the existing file's first line doesn't match `fields`, the file
    is rotated aside (`.stale-<utc>`) and started fresh — this prevents
    silent corruption when an older release (or an eval script with a
    different schema) wrote to the same path. Caller already holds
    `_LOCK` so the rename is race-safe.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_header = ",".join(fields)
    needs_init = not path.exists()
    if not needs_init:
        try:
            with path.open("r", encoding="utf-8") as rf:
                actual_header = (rf.readline() or "").rstrip("\r\n")
            if actual_header != expected_header:
                from datetime import datetime, timezone
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                rotated = path.with_suffix(path.suffix + f".stale-{stamp}")
                path.rename(rotated)
                needs_init = True
        except OSError:
            needs_init = True
    f = path.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fields)
    if needs_init:
        w.writeheader()
    return f, w


def record_result(
    result: OutputRiskResult,
    *,
    run_id: str,
    case_id: str = "",
    target_id: str = "",
    metrics_path: Optional[Path] = None,
    explain_path: Optional[Path] = None,
) -> None:
    """Append both CSVs atomically under a single lock so readers never see
    a half-updated pair."""
    m_path = metrics_path or METRICS_PATH
    e_path = explain_path or EXPLAIN_PATH

    metrics_row: Dict[str, Any] = {
        "run_id": run_id,
        "case_id": case_id,
        "target_id": target_id,
        "score": round(result.score, 4),
        "decision": result.decision,
        "output_chars": result.output_chars,
        "latency_ms": result.latency_ms,
        "flag_pii":                   int("pii" in result.flags),
        "flag_api_key":               int("api_key" in result.flags),
        "flag_unsafe_instruction":    int("unsafe_instruction" in result.flags),
        "flag_downstream_injection":  int("downstream_injection" in result.flags),
        "flag_redirect_to_unknown":   int("redirect_to_unknown" in result.flags),
        "evidence_top": " | ".join(result.evidence)[:200],
    }

    explain_rows: list[Dict[str, Any]] = []
    for flag_name, details in result.flags.items():
        # Each triggered rule/subtype gets its own row so filtering in the
        # dashboard is trivial (SELECT … WHERE flag_name='api_key').
        rules = details.get("rules") or list((details.get("hits") or {}).keys()) or ["-"]
        samples = details.get("samples") or details.get("unknown_hosts") or []
        for i, rule in enumerate(rules):
            sample = samples[i] if i < len(samples) else ""
            explain_rows.append({
                "run_id": run_id,
                "case_id": case_id,
                "target_id": target_id,
                "flag_name": flag_name,
                "rule_or_subtype": str(rule),
                "sample": str(sample)[:200],
            })

    with _LOCK:
        f1, w1 = _ensure_writer(m_path, _METRICS_FIELDS)
        try:
            w1.writerow(metrics_row)
        finally:
            f1.close()
        if explain_rows:
            f2, w2 = _ensure_writer(e_path, _EXPLAIN_FIELDS)
            try:
                w2.writerows(explain_rows)
            finally:
                f2.close()


__all__ = ["record_result", "METRICS_PATH", "EXPLAIN_PATH"]
