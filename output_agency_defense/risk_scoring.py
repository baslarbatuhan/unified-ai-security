"""
output_agency_defense/risk_scoring.py
=======================================
Tool call risk scoring for Excessive Agency guard.

Purpose:
    - Produce risk score for each tool call
    - Factors: unauthorized access attempt, sequential probing, invalid resource references
    - Risk score normalized 0.0 to 1.0
    - Output compatible with ModuleRisk for fusion gateway

Risk Factors:
    1. unauthorized_access:    Owner mismatch (IDOR attempt) → high risk
    2. sequential_probing:     Anti-enum detection → very high risk
    3. invalid_resource:       Unregistered tool or missing resource → medium risk
    4. role_violation:         User role not in allowed_roles → medium risk

Integration:
    secure_tool_wrapper calls this to produce ModuleRisk before/after execution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Any

try:
    from output_agency_defense.object_authz_guard import AuthzResult
    from output_agency_defense.anti_enum_guard import EnumGuardResult
except ImportError:
    from object_authz_guard import AuthzResult
    from anti_enum_guard import EnumGuardResult


Decision = Literal["allow", "sanitize", "block", "flag"]


# ---------------------------------------------------------------------------
# Risk weights
# ---------------------------------------------------------------------------
RISK_WEIGHTS = {
    "unauthorized_access": 0.90,
    "sequential_probing": 1.00,
    "invalid_resource": 0.60,
    "role_violation": 0.70,
    "clean": 0.00,
}


# ---------------------------------------------------------------------------
# Thresholds (from configs/secure_balanced.yaml)
# ---------------------------------------------------------------------------
@dataclass
class AgencyRiskThresholds:
    allow_max: float = 0.30
    sanitize_max: float = 0.60
    block_min: float = 0.85

    def decide(self, score: float) -> Decision:
        if score < self.allow_max:
            return "allow"
        elif score < self.sanitize_max:
            return "sanitize"
        elif score >= self.block_min:
            return "block"
        else:
            return "flag"


# ---------------------------------------------------------------------------
# ModuleRisk-compatible output
# ---------------------------------------------------------------------------
@dataclass
class AgencyModuleRisk:
    """Risk output for the output_agency module."""
    module: str = "output_agency"
    risk_score: float = 0.0
    confidence: float = 0.0
    decision: Decision = "allow"
    evidence: List[str] = field(default_factory=list)
    latency_ms: Optional[int] = None

    # Agency-specific
    risk_factors: List[str] = field(default_factory=list)
    user_id: str = ""
    tool: str = ""
    resource_id: str = ""

    def to_module_risk_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "risk_score": round(self.risk_score, 4),
            "confidence": round(self.confidence, 4),
            "decision": self.decision,
            "evidence": self.evidence,
            "latency_ms": self.latency_ms,
        }


# ---------------------------------------------------------------------------
# Agency Risk Scorer
# ---------------------------------------------------------------------------
class AgencyRiskScorer:
    """
    Computes risk score for tool calls based on multiple signals.

    Combines:
    - Authorization result (IDOR check)
    - Anti-enumeration result (probing check)
    - Tool validity (registered or not)
    - Role check (user role allowed or not)
    """

    def __init__(self, thresholds: Optional[AgencyRiskThresholds] = None):
        self.thresholds = thresholds or AgencyRiskThresholds()

    def score(
        self,
        user_id: str,
        tool_name: str,
        resource_id: str = "",
        authz_result: Optional[AuthzResult] = None,
        enum_result: Optional[EnumGuardResult] = None,
        tool_registered: bool = True,
        role_allowed: bool = True,
    ) -> AgencyModuleRisk:
        """
        Compute agency risk score from multiple signals.

        The highest risk factor dominates (max, not average).
        """
        t0 = time.time()

        risk_factors = []
        evidence = []
        scores = []

        # Factor 1: Authorization (IDOR)
        if authz_result and not authz_result.is_allowed:
            risk_factors.append("unauthorized_access")
            scores.append(RISK_WEIGHTS["unauthorized_access"])
            evidence.extend(authz_result.evidence)

        # Factor 2: Enumeration detection
        if enum_result and enum_result.is_enumeration:
            risk_factors.append("sequential_probing")
            scores.append(RISK_WEIGHTS["sequential_probing"])
            evidence.extend(enum_result.evidence)

        # Factor 3: Invalid resource / unregistered tool
        if not tool_registered:
            risk_factors.append("invalid_resource")
            scores.append(RISK_WEIGHTS["invalid_resource"])
            evidence.append(f"Tool '{tool_name}' is not registered")

        # Factor 4: Role violation
        if not role_allowed:
            risk_factors.append("role_violation")
            scores.append(RISK_WEIGHTS["role_violation"])
            evidence.append(f"User role not authorized for tool '{tool_name}'")

        # Clean call
        if not risk_factors:
            risk_factors.append("clean")
            evidence.append(f"Tool '{tool_name}' executed — all checks passed")

        # Risk score: take maximum (worst-case signal)
        risk_score = max(scores) if scores else 0.0

        # Decision
        decision = self.thresholds.decide(risk_score)

        # Confidence
        if len(risk_factors) > 1 and "clean" not in risk_factors:
            confidence = 0.95  # multiple bad signals
        elif risk_factors and "clean" not in risk_factors:
            confidence = 0.85
        else:
            confidence = 0.90  # confident it's clean

        latency_ms = int((time.time() - t0) * 1000)

        return AgencyModuleRisk(
            risk_score=round(risk_score, 4),
            confidence=round(confidence, 2),
            decision=decision,
            evidence=evidence,
            latency_ms=latency_ms,
            risk_factors=risk_factors,
            user_id=user_id,
            tool=tool_name,
            resource_id=resource_id,
        )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    from object_authz_guard import AuthzResult
    from anti_enum_guard import EnumGuardResult

    scorer = AgencyRiskScorer()

    print(f"{'='*60}")
    print(f"  AGENCY RISK SCORING DEMO")
    print(f"{'='*60}")

    # Scenario 1: Clean call
    risk = scorer.score("user_alice", "get_order", "ORD-001")
    print(f"\n  [CLEAN] {json.dumps(risk.to_module_risk_dict(), indent=4)}")

    # Scenario 2: IDOR attempt
    authz_denied = AuthzResult(
        decision="deny", resource_type="order", resource_id="ORD-001",
        user="user_bob", owner="user_alice",
        evidence=["Owner mismatch: 'user_alice' vs 'user_bob'", "IDOR detected"]
    )
    risk = scorer.score("user_bob", "get_order", "ORD-001", authz_result=authz_denied)
    print(f"\n  [IDOR] {json.dumps(risk.to_module_risk_dict(), indent=4)}")

    # Scenario 3: Enumeration detected
    enum_detected = EnumGuardResult(
        decision="block", risk_score=1.0, user_id="attacker",
        resource_id="ORD-1005", is_enumeration=True, sequential_count=5,
        unique_attempts=5, evidence=["enumeration_detected", "5 sequential IDs"]
    )
    risk = scorer.score("attacker", "get_order", "ORD-1005", enum_result=enum_detected)
    print(f"\n  [ENUM] {json.dumps(risk.to_module_risk_dict(), indent=4)}")

    # Scenario 4: Unregistered tool
    risk = scorer.score("user_alice", "delete_database", "", tool_registered=False)
    print(f"\n  [UNREG] {json.dumps(risk.to_module_risk_dict(), indent=4)}")

    # Scenario 5: Combined (IDOR + enumeration)
    risk = scorer.score("attacker", "get_order", "ORD-1006",
                        authz_result=authz_denied, enum_result=enum_detected)
    print(f"\n  [COMBINED] risk_factors={risk.risk_factors}")
    print(f"             {json.dumps(risk.to_module_risk_dict(), indent=4)}")

    print(f"\n{'='*60}")
