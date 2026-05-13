"""configs/timeout_loader.py
=============================
Tiny typed loader for `timeout_config.yaml` and `service_limits.yaml`.

Kept as a module (not a class) so importers stay cheap and multiple
gateways / runners can share the same parsed profile without globals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_CONFIGS_DIR = Path(__file__).resolve().parent
_DEFAULT_TIMEOUT_PATH = _CONFIGS_DIR / "timeout_config.yaml"
_DEFAULT_LIMITS_PATH = _CONFIGS_DIR / "service_limits.yaml"


def load_timeout_profile(
    profile: str = "standard",
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return a single profile dict. Raises KeyError if the profile name
    is unknown — caller should surface this as a config error."""
    path = path or _DEFAULT_TIMEOUT_PATH
    if not path.exists():
        raise FileNotFoundError(f"timeout config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    profiles = data.get("profiles") or {}
    if profile not in profiles:
        raise KeyError(
            f"timeout profile {profile!r} missing; available: {sorted(profiles)}"
        )
    return profiles[profile]


def load_service_limits(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or _DEFAULT_LIMITS_PATH
    if not path.exists():
        raise FileNotFoundError(f"service limits not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def module_budget_ms(profile: Dict[str, Any], module: str) -> int:
    """Lookup helper; returns 0 if the module is absent (callers treat 0
    as 'no budget enforced')."""
    return int((profile.get("modules") or {}).get(module, 0))


# ---------------------------------------------------------------------------
# Hafta 11: fail-CLOSED policy and per-call sub-budgets
# ---------------------------------------------------------------------------

# Policy → (risk_score, decision) when a module times out or raises.
# `allow` is the legacy fail-OPEN behaviour and is kept for backward compat;
# callers should pick `sanitize` or `block` in production profiles.
_POLICY_RISK: Dict[str, float] = {
    "allow": 0.0,
    "sanitize": 0.5,
    "block": 1.0,
}

_VALID_POLICIES = frozenset(_POLICY_RISK.keys())


def on_timeout_policy(profile: Dict[str, Any], module: str, default: str = "block") -> str:
    """Return the on-timeout decision for a module, e.g. 'block'.

    Returns `default` (fail-CLOSED) when the profile doesn't specify a
    policy for the module — never silently fail-open unless explicitly
    configured. The caller maps this through `policy_risk_score()` to
    get the synthetic `risk_score` that the engine injects in the
    fail-CLOSED ModuleRisk.
    """
    policy_map = profile.get("on_timeout") or {}
    pol = str(policy_map.get(module, default)).lower().strip()
    if pol not in _VALID_POLICIES:
        # Defensive: unknown policy reverts to default rather than crashing
        # the request path. Keeps misconfig from taking the gateway down.
        return default
    return pol


def policy_risk_score(policy: str) -> float:
    """Map a policy string to a synthetic risk_score in [0.0, 1.0].

    `block` → 1.0, `sanitize` → 0.5, `allow` → 0.0. Unknown policies
    default to 1.0 (fail-CLOSED) — never silently fail-open."""
    return _POLICY_RISK.get(policy, 1.0)


def llm_judge_budget_ms(profile: Dict[str, Any], default_ms: int = 18000) -> int:
    """LLM judge per-call ceiling. Falls back to `default_ms` when the
    profile is missing the key (older configs predate this field)."""
    val = profile.get("llm_judge_ms")
    if val is None:
        return int(default_ms)
    try:
        return int(val)
    except (TypeError, ValueError):
        return int(default_ms)


__all__ = [
    "load_timeout_profile",
    "load_service_limits",
    "module_budget_ms",
    "on_timeout_policy",
    "policy_risk_score",
    "llm_judge_budget_ms",
]
