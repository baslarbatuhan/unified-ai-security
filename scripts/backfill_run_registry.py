"""scripts/backfill_run_registry.py
====================================
One-shot migration: register every run in the legacy aggregate CSV.

Reads `runs/external_eval_results.csv`, groups rows by `run_id`, and
materialises the per-run artefacts that newer code expects:

    runs/<run_id>/results.csv     — rows scoped to this run
    runs/<run_id>/manifest.json   — schema_version + provenance
    runs/_registry.jsonl          — appended one entry per run

Idempotent: skips runs that already have a manifest, and dedupes
registry entries by run_id (so re-running is safe).

Run from project root:
    .venv/bin/python scripts/backfill_run_registry.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.run_manifest import (  # noqa: E402
    REGISTRY_FILENAME,
    MANIFEST_FILENAME,
    write_results_csv,
    write_manifest,
    append_registry_entry,
    read_registry,
)

_RUNS_DIR = _PROJECT_ROOT / "runs"
_AGG_CSV = _RUNS_DIR / "external_eval_results.csv"


def _read_status(run_dir: Path) -> Dict[str, Optional[str]]:
    """Best-effort read of started_at / ended_at / exit_code from status.json."""
    p = run_dir / "status.json"
    if not p.exists():
        return {"started_at": None, "ended_at": None, "exit_code": None}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return {
            "started_at": d.get("started_at"),
            "ended_at": d.get("ended_at") or d.get("updated_at"),
            "exit_code": d.get("exit_code"),
        }
    except (json.JSONDecodeError, OSError):
        return {"started_at": None, "ended_at": None, "exit_code": None}


def main() -> int:
    if not _AGG_CSV.exists():
        print(f"[backfill] no aggregate CSV at {_AGG_CSV} — nothing to backfill")
        return 0

    # Existing registry → set of already-known run_ids (skip dupes).
    known = {entry.get("run_id") for entry in read_registry(_RUNS_DIR)}
    print(f"[backfill] registry already knows {len(known)} run(s)")

    # Group rows by run_id, preserving column order.
    with _AGG_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames: List[str] = list(reader.fieldnames or [])
        groups: Dict[str, List[Dict[str, str]]] = {}
        for row in reader:
            rid = (row.get("run_id") or "").strip()
            if not rid:
                continue
            groups.setdefault(rid, []).append(row)

    print(f"[backfill] aggregate CSV has {len(groups)} distinct run_id(s)")

    migrated = 0
    skipped_manifest = 0
    skipped_registry = 0
    for run_id, rows in groups.items():
        run_dir = _RUNS_DIR / run_id
        # Don't clobber existing per-run artefacts — manifest presence is
        # the canonical "this run is already migrated" signal.
        if (run_dir / MANIFEST_FILENAME).exists():
            skipped_manifest += 1
            if run_id not in known:
                # Manifest exists but registry entry is missing — fix it.
                first = rows[0]
                status = _read_status(run_dir)
                append_registry_entry(
                    _RUNS_DIR,
                    run_id=run_id,
                    target_id=first.get("target_id") or None,
                    suite=first.get("suite") or None,
                    started_at=status["started_at"],
                    ended_at=status["ended_at"],
                    exit_code=status["exit_code"] if isinstance(status["exit_code"], int) else None,
                    n_cases=len(rows),
                    n_rows=len(rows),
                    relative_path=str(run_dir.relative_to(_PROJECT_ROOT)),
                )
                known.add(run_id)
            continue

        # Derive run-level fields from the first row (all rows in a group
        # share target_id and suite by construction).
        first = rows[0]
        target_id = first.get("target_id") or None
        suite = first.get("suite") or None

        status = _read_status(run_dir)

        write_results_csv(run_dir, rows, fieldnames)
        write_manifest(
            run_dir,
            run_id=run_id,
            target_id=target_id,
            suite=suite,
            started_at=status["started_at"],
            ended_at=status["ended_at"],
            exit_code=status["exit_code"] if isinstance(status["exit_code"], int) else None,
            n_cases=len(rows),
            n_rows=len(rows),
            sources={"external_eval_results": _AGG_CSV.name},
            extra={"backfilled": True},
        )
        if run_id in known:
            skipped_registry += 1
        else:
            append_registry_entry(
                _RUNS_DIR,
                run_id=run_id,
                target_id=target_id,
                suite=suite,
                started_at=status["started_at"],
                ended_at=status["ended_at"],
                exit_code=status["exit_code"] if isinstance(status["exit_code"], int) else None,
                n_cases=len(rows),
                n_rows=len(rows),
                relative_path=str(run_dir.relative_to(_PROJECT_ROOT)),
            )
            known.add(run_id)
        migrated += 1

    print(
        f"[backfill] migrated={migrated}  "
        f"skipped_manifest_exists={skipped_manifest}  "
        f"skipped_registry_exists={skipped_registry}"
    )
    print(f"[backfill] registry: {_RUNS_DIR / REGISTRY_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
