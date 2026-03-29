"""
output_agency_defense/object_authz_guard.py
=============================================
Object-level authorization guard to prevent IDOR attacks.

Purpose:
    - authorize(resource_type, resource_id, session) function
    - Steps: find resource → get owner → compare with session.user
    - Owner != session.user → access denied
    - Users can access their own resources, not others'

OWASP Reference:
    - IDOR Prevention: verify user permission for every object access
    - BOLA (Broken Object Level Authorization): OWASP API Security #1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

try:
    from output_agency_defense.resource_registry import ResourceRegistry
    from output_agency_defense.error_policy import uniform_error
except ImportError:
    from resource_registry import ResourceRegistry
    from error_policy import uniform_error


# ---------------------------------------------------------------------------
# Session model
# ---------------------------------------------------------------------------
@dataclass
class Session:
    """Represents an authenticated user session."""
    user: str
    role: str = "basic"
    permissions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Authorization result
# ---------------------------------------------------------------------------
AuthzDecision = Literal["allow", "deny"]


@dataclass
class AuthzResult:
    """Result of an authorization check."""
    decision: AuthzDecision
    resource_type: str
    resource_id: str
    user: str
    owner: Optional[str] = None
    reason: str = ""
    evidence: List[str] = field(default_factory=list)

    @property
    def is_allowed(self) -> bool:
        return self.decision == "allow"


# ---------------------------------------------------------------------------
# Authorization Guard
# ---------------------------------------------------------------------------
class ObjectAuthzGuard:
    """
    Enforces object-level authorization for every tool call.

    Flow:
        1. Look up resource via registry
        2. Get owner from resource record
        3. Compare owner with session.user
        4. If mismatch → deny (returns uniform error)

    This prevents IDOR attacks where user_A tries to
    access user_B's resources by manipulating resource_id.
    """

    def __init__(self, registry: ResourceRegistry):
        self.registry = registry

    def authorize(
        self,
        resource_type: str,
        resource_id: str,
        session: Session,
    ) -> AuthzResult:
        """
        Check if session.user is authorized to access the resource.

        Args:
            resource_type: Type of resource ("order", "ticket", etc.)
            resource_id:   ID of the specific resource
            session:       Current user session

        Returns:
            AuthzResult with decision and evidence.
        """
        evidence = []

        # Step 1: Check if resource type is registered
        if not self.registry.is_registered(resource_type):
            evidence.append(f"Resource type '{resource_type}' not registered")
            return AuthzResult(
                decision="deny",
                resource_type=resource_type,
                resource_id=resource_id,
                user=session.user,
                reason=uniform_error(),
                evidence=evidence,
            )

        # Step 2: Find the resource
        resource = self.registry.find(resource_type, resource_id)
        if resource is None:
            # Resource not found — return SAME error as unauthorized
            # This prevents resource enumeration attacks
            evidence.append(f"Resource {resource_type}/{resource_id} not found")
            return AuthzResult(
                decision="deny",
                resource_type=resource_type,
                resource_id=resource_id,
                user=session.user,
                reason=uniform_error(),
                evidence=evidence,
            )

        # Step 3: Get owner
        owner = self.registry.get_owner(resource_type, resource_id)
        if owner is None:
            evidence.append(f"Cannot determine owner for {resource_type}/{resource_id}")
            return AuthzResult(
                decision="deny",
                resource_type=resource_type,
                resource_id=resource_id,
                user=session.user,
                reason=uniform_error(),
                evidence=evidence,
            )

        # Step 4: Compare owner with session user
        if owner != session.user:
            # IDOR ATTEMPT DETECTED
            evidence.append(f"Owner mismatch: resource owned by '{owner}', requested by '{session.user}'")
            evidence.append("Potential IDOR attack detected")
            return AuthzResult(
                decision="deny",
                resource_type=resource_type,
                resource_id=resource_id,
                user=session.user,
                owner=owner,
                reason=uniform_error(),
                evidence=evidence,
            )

        # Access granted
        evidence.append(f"Owner '{owner}' matches session user '{session.user}'")
        return AuthzResult(
            decision="allow",
            resource_type=resource_type,
            resource_id=resource_id,
            user=session.user,
            owner=owner,
            reason="access_granted",
            evidence=evidence,
        )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from resource_registry import create_demo_registry

    registry = create_demo_registry()
    guard = ObjectAuthzGuard(registry)

    # Simulate sessions
    alice = Session(user="user_alice", role="basic")
    bob = Session(user="user_bob", role="basic")

    print(f"{'='*60}")
    print(f"  OBJECT-LEVEL AUTHORIZATION DEMO")
    print(f"{'='*60}")

    test_cases = [
        # (session, resource_type, resource_id, expected)
        (alice, "order", "ORD-001", "ALLOW — Alice's own order"),
        (bob,   "order", "ORD-001", "DENY  — Bob trying Alice's order (IDOR)"),
        (bob,   "order", "ORD-002", "ALLOW — Bob's own order"),
        (alice, "ticket", "TKT-101", "ALLOW — Alice's own ticket"),
        (bob,   "ticket", "TKT-101", "DENY  — Bob trying Alice's ticket (IDOR)"),
        (alice, "order", "ORD-999", "DENY  — Non-existent order"),
        (alice, "invoice", "INV-01", "DENY  — Unregistered resource type"),
    ]

    for session, rtype, rid, description in test_cases:
        result = guard.authorize(rtype, rid, session)
        status = "ALLOW" if result.is_allowed else "DENY"
        print(f"\n  [{status}] {session.user} → {rtype}/{rid}")
        print(f"    Expected: {description}")
        print(f"    Reason:   {result.reason}")
        for e in result.evidence:
            print(f"    Evidence: {e}")
