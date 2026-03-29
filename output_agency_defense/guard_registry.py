"""
output_agency_defense/guard_registry.py
=========================================
Registry of active security guards.

Purpose:
    - Track which guards are active in the system
    - Enforce required guards: object_authz, anti_enum
    - System must not run without required guards
    - Used by security_selfcheck and startup validation

Required Guards:
    - object_authz: Object-level authorization (IDOR prevention)
    - anti_enum: Anti-enumeration (ID brute-force prevention)

Usage:
    registry = GuardRegistry()
    registry.register("object_authz", authz_guard)
    registry.register("anti_enum", enum_guard)
    registry.validate()  # raises if required guards missing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Required guards
# ---------------------------------------------------------------------------
REQUIRED_GUARDS = ["object_authz", "anti_enum"]


# ---------------------------------------------------------------------------
# Guard entry
# ---------------------------------------------------------------------------
@dataclass
class GuardEntry:
    name: str
    instance: Any
    enabled: bool = True
    description: str = ""
    required: bool = False


# ---------------------------------------------------------------------------
# Guard Registry
# ---------------------------------------------------------------------------
class GuardRegistry:
    """
    Central registry for all active security guards.

    Enforces that required guards (object_authz, anti_enum)
    are registered and enabled before the system can operate.
    """

    def __init__(self):
        self._guards: Dict[str, GuardEntry] = {}

    def register(
        self,
        name: str,
        instance: Any,
        enabled: bool = True,
        description: str = "",
    ) -> None:
        """Register a security guard."""
        self._guards[name] = GuardEntry(
            name=name,
            instance=instance,
            enabled=enabled,
            description=description,
            required=name in REQUIRED_GUARDS,
        )

    def unregister(self, name: str) -> bool:
        if name in self._guards:
            del self._guards[name]
            return True
        return False

    def is_registered(self, name: str) -> bool:
        return name in self._guards

    def is_enabled(self, name: str) -> bool:
        entry = self._guards.get(name)
        return entry is not None and entry.enabled

    def get(self, name: str) -> Optional[Any]:
        """Get guard instance by name."""
        entry = self._guards.get(name)
        return entry.instance if entry and entry.enabled else None

    def enable(self, name: str) -> bool:
        if name in self._guards:
            self._guards[name].enabled = True
            return True
        return False

    def disable(self, name: str) -> bool:
        if name in self._guards:
            self._guards[name].enabled = False
            return True
        return False

    def list_guards(self) -> List[Dict]:
        """List all registered guards with status."""
        return [
            {
                "name": g.name,
                "enabled": g.enabled,
                "required": g.required,
                "description": g.description,
            }
            for g in self._guards.values()
        ]

    def list_active(self) -> List[str]:
        """List names of active (enabled) guards."""
        return [g.name for g in self._guards.values() if g.enabled]

    def list_missing_required(self) -> List[str]:
        """List required guards that are not registered or not enabled."""
        missing = []
        for name in REQUIRED_GUARDS:
            if not self.is_enabled(name):
                missing.append(name)
        return missing

    def validate(self) -> Dict:
        """
        Validate that all required guards are active.

        Returns:
            {"valid": bool, "missing": List[str], "active": List[str]}

        Raises:
            RuntimeError if required guards are missing.
        """
        missing = self.list_missing_required()
        active = self.list_active()

        result = {
            "valid": len(missing) == 0,
            "missing": missing,
            "active": active,
            "total_registered": len(self._guards),
        }

        if missing:
            raise RuntimeError(
                f"security_guards_missing: Required guards not active: {missing}. "
                f"System cannot operate without: {REQUIRED_GUARDS}"
            )

        return result


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    registry = GuardRegistry()

    print(f"{'='*55}")
    print(f"  GUARD REGISTRY DEMO")
    print(f"  Required guards: {REQUIRED_GUARDS}")
    print(f"{'='*55}")

    # Test 1: Missing required guards
    print(f"\n  [Test 1] No guards registered:")
    try:
        registry.validate()
    except RuntimeError as e:
        print(f"    ERROR: {e}")

    # Test 2: Partial registration
    print(f"\n  [Test 2] Only object_authz registered:")
    registry.register("object_authz", "mock_authz_instance", description="IDOR guard")
    try:
        registry.validate()
    except RuntimeError as e:
        print(f"    ERROR: {e}")

    # Test 3: All required guards
    print(f"\n  [Test 3] All required guards registered:")
    registry.register("anti_enum", "mock_enum_instance", description="Anti-enumeration guard")
    result = registry.validate()
    print(f"    Valid: {result['valid']}")
    print(f"    Active: {result['active']}")

    # Test 4: Disable a required guard
    print(f"\n  [Test 4] Disable anti_enum:")
    registry.disable("anti_enum")
    try:
        registry.validate()
    except RuntimeError as e:
        print(f"    ERROR: {e}")

    # Test 5: Full listing
    registry.enable("anti_enum")
    registry.register("rate_limiter", "mock_rate_limiter", description="Optional rate limiter")
    print(f"\n  [Test 5] All guards:")
    for g in registry.list_guards():
        req = " [REQUIRED]" if g["required"] else ""
        status = "ACTIVE" if g["enabled"] else "DISABLED"
        print(f"    {g['name']:20s} | {status:8s}{req} | {g['description']}")

    print(f"\n{'='*55}")
