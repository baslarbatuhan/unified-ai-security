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


__all__ = [
    "load_timeout_profile",
    "load_service_limits",
    "module_budget_ms",
]
