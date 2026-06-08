"""external_eval/secret_resolver.py
======================================
Single helper that resolves an auth field's ``*_env`` reference to its
actual value at request time.

Lookup order
------------
1. ``os.environ[name]`` — production-friendly: real secret managers
   (Kubernetes secrets, Doppler, AWS Parameter Store) inject env-vars
   at container boot.
2. ``secrets_store.get(name)`` — local, dashboard-managed vault for
   single-machine dev/eval where editing ``.env`` and restarting the
   container is overkill.
3. Empty string — caller decides what to do (api_adapter surfaces an
   empty Authorization header so the downstream 401 is the visible
   failure mode).

This priority preserves behaviour for every existing deployment: tests
that set ``os.environ`` keep working unchanged, production keeps reading
its real secret store, and laptops with nothing in env-vars fall through
to the vault.
"""

from __future__ import annotations

import os
import re

from external_eval import secrets_store


# ``${NAME}`` placeholder — POSIX env-var name rules. The pattern is
# anchored on the braces so a bare ``$FOO`` stays literal (rare but
# legal in custom header values).  Module-level so both api_adapter
# (header values) and web_adapter (cookie values) can reuse it via
# ``interpolate()``.
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def resolve(env_name: str) -> str:
    """Resolve ``env_name`` to a string value. Empty if nowhere found.

    Treats ``""`` (empty string in os.environ) the same as "not set" so
    a placeholder like ``MY_KEY=`` in .env doesn't shadow a vault entry.
    """
    if not env_name:
        return ""
    val = os.environ.get(env_name, "")
    if val:
        return val
    stored = secrets_store.get(env_name)
    return stored or ""


def source_of(env_name: str) -> str:
    """Diagnostic helper: where does the value currently come from?

    Returns ``"env"``, ``"vault"``, or ``"missing"``. Used by the
    dashboard form's status indicator so the user knows whether their
    paste landed in the vault or whether the env-var is masking it.
    """
    if not env_name:
        return "missing"
    if os.environ.get(env_name, ""):
        return "env"
    if secrets_store.has(env_name) and (secrets_store.get(env_name) or ""):
        return "vault"
    return "missing"


def interpolate(value):
    """Replace every ``${NAME}`` in ``value`` with ``resolve(NAME)``.

    Multiple placeholders in one string are all substituted
    (``"scheme=${PFX}/${TOK}"`` works).  Unresolved names become empty
    strings — same contract as ``resolve``, so the downstream 401/403
    stays visible.  Non-string inputs (defensive: dict.values() may
    smuggle non-strings past Pydantic via dict-merge) are returned
    as-is.
    """
    if not isinstance(value, str) or "${" not in value:
        return value
    return _VAR_RE.sub(lambda m: resolve(m.group(1)), value)


def iter_var_names(value):
    """Yield each ``NAME`` referenced as ``${NAME}`` in ``value``.

    Used by diagnostics paths (``routes_targets._collect_auth_sources``)
    that need to enumerate references without performing resolution.
    """
    if not isinstance(value, str):
        return
    for m in _VAR_RE.finditer(value):
        yield m.group(1)


__all__ = ["resolve", "source_of", "interpolate", "iter_var_names"]
