"""external_eval/target_loader.py
==================================
Loader + CRUD helpers for `external_eval/targets.yaml`.

The file itself is the source of truth. The dashboard mutates it through
these helpers; every mutation goes through Pydantic validation so we can
never end up with a half-valid targets.yaml on disk.

Concurrency note
----------------
Dashboard + API workers may write concurrently. We hold an exclusive file
lock (`fcntl.LOCK_EX` on POSIX, best-effort on Windows) for the duration
of a read-modify-write, but this is per-process — for multi-worker
deployments, front the dashboard API with a single writer process.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import RLock
from typing import List, Optional

import yaml

from schemas.target_schema import TargetConfig, TargetsFile

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGETS_PATH = _PROJECT_ROOT / "external_eval" / "targets.yaml"

# Serialize writes within a process. Cross-process coordination is the
# caller's job (see module docstring).
_FILE_LOCK = RLock()


# ---------------------------------------------------------------------------
# Core read / write
# ---------------------------------------------------------------------------
def load_targets(path: Optional[Path] = None) -> TargetsFile:
    """Load + validate the targets file. Missing file returns an empty
    `TargetsFile` so first-time bootstrap works."""
    path = path or DEFAULT_TARGETS_PATH
    if not path.exists():
        return TargetsFile()
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return TargetsFile.model_validate(raw)


def save_targets(tf: TargetsFile, path: Optional[Path] = None) -> Path:
    """Write the targets file atomically (write to `.tmp` then rename)."""
    path = path or DEFAULT_TARGETS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = tf.model_dump(mode="json", exclude_none=True)
    with _FILE_LOCK:
        with tmp.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
        os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# CRUD convenience
# ---------------------------------------------------------------------------
def get_target(target_id: str, path: Optional[Path] = None) -> Optional[TargetConfig]:
    for t in load_targets(path).targets:
        if t.id == target_id:
            return t
    return None


def list_targets(path: Optional[Path] = None, *, enabled_only: bool = False) -> List[TargetConfig]:
    targets = load_targets(path).targets
    if enabled_only:
        targets = [t for t in targets if t.enabled]
    return targets


def upsert_target(target: TargetConfig, path: Optional[Path] = None) -> Path:
    """Insert or overwrite a target by id. Validated end-to-end."""
    path = path or DEFAULT_TARGETS_PATH
    with _FILE_LOCK:
        tf = load_targets(path)
        tf.targets = [t for t in tf.targets if t.id != target.id] + [target]
        # Re-validate the whole file (uniqueness, cross-field rules).
        tf = TargetsFile.model_validate(tf.model_dump())
        return save_targets(tf, path)


def delete_target(target_id: str, path: Optional[Path] = None) -> bool:
    """Return True if a target was removed."""
    path = path or DEFAULT_TARGETS_PATH
    with _FILE_LOCK:
        tf = load_targets(path)
        before = len(tf.targets)
        tf.targets = [t for t in tf.targets if t.id != target_id]
        if len(tf.targets) == before:
            return False
        save_targets(tf, path)
        return True


__all__ = [
    "DEFAULT_TARGETS_PATH",
    "load_targets",
    "save_targets",
    "get_target",
    "list_targets",
    "upsert_target",
    "delete_target",
]
