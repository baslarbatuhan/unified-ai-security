"""
output_agency_defense/anti_enum_guard.py
==========================================
Anti-enumeration guard for preventing ID brute-force attacks.

Purpose:
    - Block users making too many unique resource_id attempts
    - Track: unique_id_attempts_per_minute, sequential_id_delta
    - When threshold exceeded: decision=block, risk_score=1.0, evidence=enumeration_detected

OWASP Reference:
    - IDOR Prevention: don't allow resource enumeration
    - LLM06:2025 Excessive Agency: limit autonomous probing

Integration:
    Called by secure_tool_wrapper BEFORE authorization check.
    If probing detected → immediately block, skip authz.

Usage:
    guard = AntiEnumGuard()
    result = guard.check("user_bob", "ORD-1001")
    # After many sequential attempts:
    # result.decision == "block", result.risk_score == 1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

try:
    from output_agency_defense.sequential_probe_detector import SequentialProbeDetector, ProbeResult
except ImportError:
    from sequential_probe_detector import SequentialProbeDetector, ProbeResult


Decision = Literal["allow", "block"]


# ---------------------------------------------------------------------------
# Guard result
# ---------------------------------------------------------------------------
@dataclass
class EnumGuardResult:
    """Result of anti-enumeration check."""
    decision: Decision
    risk_score: float
    user_id: str
    resource_id: str
    is_enumeration: bool = False
    sequential_count: int = 0
    unique_attempts: int = 0
    evidence: List[str] = field(default_factory=list)

    @property
    def is_blocked(self) -> bool:
        return self.decision == "block"


# ---------------------------------------------------------------------------
# Anti-Enumeration Guard
# ---------------------------------------------------------------------------
class AntiEnumGuard:
    """
    Guards against ID enumeration attacks.

    Wraps SequentialProbeDetector and produces block/allow decisions.
    When enumeration is detected:
        decision = block
        risk_score = 1.0
        evidence = ["enumeration_detected"]

    Tracked metrics:
        - unique_id_attempts_per_minute
        - sequential_id_delta
    """

    def __init__(
        self,
        window_seconds: int = 60,
        seq_threshold: int = 3,
        rate_threshold: int = 10,
    ):
        self.detector = SequentialProbeDetector(
            window_seconds=window_seconds,
            seq_threshold=seq_threshold,
            rate_threshold=rate_threshold,
        )

    def check(self, user_id: str, resource_id: str) -> EnumGuardResult:
        """
        Check if a resource access attempt constitutes enumeration.

        Args:
            user_id:     The user making the attempt
            resource_id: The resource ID being accessed

        Returns:
            EnumGuardResult with block/allow decision.
        """
        probe = self.detector.record_attempt(user_id, resource_id)

        if probe.is_probing:
            evidence = ["enumeration_detected"] + probe.evidence
            return EnumGuardResult(
                decision="block",
                risk_score=1.0,
                user_id=user_id,
                resource_id=resource_id,
                is_enumeration=True,
                sequential_count=probe.sequential_count,
                unique_attempts=probe.unique_attempts_in_window,
                evidence=evidence,
            )

        return EnumGuardResult(
            decision="allow",
            risk_score=probe.risk_contribution,
            user_id=user_id,
            resource_id=resource_id,
            is_enumeration=False,
            sequential_count=probe.sequential_count,
            unique_attempts=probe.unique_attempts_in_window,
            evidence=probe.evidence,
        )

    def get_user_stats(self, user_id: str) -> dict:
        return self.detector.get_user_stats(user_id)

    def reset_user(self, user_id: str):
        self.detector.reset_user(user_id)

    def reset_all(self):
        self.detector.reset_all()


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    guard = AntiEnumGuard(window_seconds=60, seq_threshold=3, rate_threshold=10)

    print(f"{'='*60}")
    print(f"  ANTI-ENUMERATION GUARD DEMO")
    print(f"{'='*60}")

    # Normal access
    print(f"\n  [Normal access]")
    for rid in ["ORD-001", "ORD-005", "ORD-010"]:
        r = guard.check("user_alice", rid)
        print(f"    {rid}: decision={r.decision} | risk={r.risk_score:.2f}")

    # Enumeration attack
    print(f"\n  [Enumeration attack — sequential IDs]")
    guard.reset_all()
    for i in range(1001, 1008):
        rid = f"ORD-{i}"
        r = guard.check("attacker_bob", rid)
        print(f"    {rid}: decision={r.decision} | risk={r.risk_score:.2f} | seq={r.sequential_count}")
        if r.is_blocked:
            for e in r.evidence:
                print(f"      → {e}")

    # After detection, all further attempts blocked
    print(f"\n  [Post-detection — attacker tries different IDs]")
    for rid in ["ORD-2001", "ORD-3001"]:
        r = guard.check("attacker_bob", rid)
        print(f"    {rid}: decision={r.decision} | risk={r.risk_score:.2f} | unique={r.unique_attempts}")

    print(f"\n{'='*60}")
