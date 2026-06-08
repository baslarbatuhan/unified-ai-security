"""external_eval/secrets_store.py
====================================
Local, git-ignored secrets vault.

The dashboard's Targets page lets users paste API keys directly into the
form. Those values must NOT enter ``targets.yaml`` (committed) or any
container env-var declared in compose. Instead they live here, in a
single YAML file under ``runs/.secrets.yaml`` (``runs/`` is already in
.gitignore and bind-mounted into the gateway container, so the vault
survives ``docker compose down/up``).

Resolution priority
-------------------
At request time, the api_adapter resolves an ``auth.token_env`` reference
through ``secret_resolver.resolve(name)`` which checks ``os.environ``
first and falls back to this vault. That ordering means production
deployments that inject secrets via Kubernetes / Vault / Doppler keep
working unchanged — the local vault only fills the gap when no env-var
is set.

File shape
----------
``runs/.secrets.yaml``::

    version: 1
    secrets:
      MY_API_KEY: "actual-secret-value"
      INTERNAL_BEARER: "..."

Threading model mirrors ``target_loader``: in-process ``RLock`` +
atomic rename on write. Cross-process coordination is the caller's
job (single gateway worker is the supported deployment).
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SECRETS_PATH = _PROJECT_ROOT / "runs" / ".secrets.yaml"

_FILE_LOCK = RLock()


def _path() -> Path:
    """Resolve the vault path, allowing tests to override via env-var.

    ``UAIS_SECRETS_PATH`` is honoured if set; otherwise the default
    ``runs/.secrets.yaml`` under the project root is used. Tests use
    ``monkeypatch.setenv`` + a tmp_path to isolate state.
    """
    override = os.environ.get("UAIS_SECRETS_PATH")
    return Path(override) if override else DEFAULT_SECRETS_PATH


def _load_raw() -> Dict[str, str]:
    """Return the raw {name: value} map. Missing file → empty dict.

    Defensive on read errors: an unreadable vault (wrong perms after a
    Docker volume mount handed it to a different uid, file locked by
    another process, etc.) degrades to an empty dict so the resolver
    falls through to "missing" rather than crashing every auth header
    build. Diagnostic output is intentionally absent — the dashboard's
    /secrets endpoint will surface the inconsistency the next time the
    user opens the form.
    """
    p = _path()
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except OSError:
        return {}
    secrets = doc.get("secrets") if isinstance(doc, dict) else None
    if not isinstance(secrets, dict):
        return {}
    return {str(k): "" if v is None else str(v) for k, v in secrets.items()}


def _save_raw(secrets: Dict[str, str]) -> None:
    """Write atomically; ensure the file is mode 0600 on POSIX so other
    users on the host can't read it. Best-effort on Windows (chmod is a
    no-op for non-owner ACLs)."""
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    payload = {"version": 1, "secrets": dict(secrets)}
    with _FILE_LOCK:
        with tmp.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=True, allow_unicode=True)
        try:
            os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            # chmod is not supported on every filesystem (e.g. Windows
            # network shares). The file still lives under runs/ which is
            # gitignored — defence-in-depth, not the only line.
            pass
        os.replace(tmp, p)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get(name: str) -> Optional[str]:
    """Return the stored value for ``name``, or None if not set.

    Empty-string values are treated as "set but blank" and returned as
    "" (not None) — callers can distinguish "user explicitly cleared
    the field" from "never set" if they care.
    """
    if not name:
        return None
    with _FILE_LOCK:
        secrets = _load_raw()
    return secrets.get(name)


def has(name: str) -> bool:
    """True iff ``name`` exists in the vault (regardless of value)."""
    return get(name) is not None


def set(name: str, value: str) -> None:  # noqa: A001 (shadow built-in by design)
    """Upsert ``name`` → ``value``. Empty value is allowed (treated as
    "stored but empty"); to remove an entry entirely use ``delete``."""
    if not name:
        raise ValueError("secret name cannot be empty")
    with _FILE_LOCK:
        secrets = _load_raw()
        secrets[str(name)] = "" if value is None else str(value)
        _save_raw(secrets)


def delete(name: str) -> bool:
    """Remove ``name`` from the vault. Returns True if a value was
    removed, False if the name was not present."""
    if not name:
        return False
    with _FILE_LOCK:
        secrets = _load_raw()
        if name not in secrets:
            return False
        del secrets[name]
        _save_raw(secrets)
        return True


def list_names() -> List[str]:
    """Sorted list of stored secret names (values never leave the
    module via this call — use ``get`` if you need a value)."""
    with _FILE_LOCK:
        secrets = _load_raw()
    return sorted(secrets.keys())


__all__ = ["get", "has", "set", "delete", "list_names", "DEFAULT_SECRETS_PATH"]
