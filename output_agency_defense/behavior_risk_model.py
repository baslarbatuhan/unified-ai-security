"""
output_agency_defense/behavior_risk_model.py
===============================================
Enhanced behavioral risk model.

Purpose:
    Combines BehaviorMonitor signals with existing agency signals
    to produce a unified behavioral risk score.

    Integrates:
    - behavior_monitor.py  → burst, diversity, repeat, failed auth, lateral movement
    - anti_enum_guard.py   → sequential ID probing
    - parameter_validation.py → malformed / malicious params

    Produces a single risk score + decision for the agency module.

Risk formula:
    weighted = 0.40*behavior + 0.35*enum + 0.25*param
    final    = max(weighted, highest_component * 0.85)
    → A single critical signal (e.g. enumeration) can trigger block alone.

Usage:
    model = BehaviorRiskModel(monitor, enum_guard, param_validator)
    result = model.assess("user_bob", "get_order", "ORD-001",
                          args={"resource_id": "ORD-001"}, was_authorized=False)
    result.to_module_risk_dict()  # → fusion gateway compatible
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

try:
    from output_agency_defense.behavior_monitor import BehaviorMonitor, BehaviorAssessment
    from output_agency_defense.anti_enum_guard import AntiEnumGuard, EnumGuardResult
    from output_agency_defense.parameter_validation import ParameterValidator, ValidationResult
except ImportError:
    from behavior_monitor import BehaviorMonitor, BehaviorAssessment
    from anti_enum_guard import AntiEnumGuard, EnumGuardResult
    from parameter_validation import ParameterValidator, ValidationResult


Decision = Literal["allow", "sanitize", "flag", "block"]

THRESHOLDS = {"allow": 0.30, "sanitize": 0.60, "block": 0.85}


def _decide(score: float) -> Decision:
    if score < THRESHOLDS["allow"]:
        return "allow"
    elif score < THRESHOLDS["sanitize"]:
        return "sanitize"
    elif score >= THRESHOLDS["block"]:
        return "block"
    else:
        return "flag"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class BehaviorRiskResult:
    """Combined behavioral risk assessment."""
    user_id: str
    tool: str
    risk_score: float = 0.0
    decision: Decision = "allow"
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    latency_ms: Optional[int] = None

    # Component scores
    behavior_score: float = 0.0
    enum_score: float = 0.0
    param_score: float = 0.0

    # Component details
    behavior_risk_level: str = "low"
    behavior_signals: List[str] = field(default_factory=list)
    enum_detected: bool = False
    param_valid: bool = True

    # --- Week 4 additions ---
    @property
    def behavior_risk_score(self) -> float:
        """Unified behavioral risk score (alias for risk_score)."""
        return self.risk_score

    @property
    def signals_triggered(self) -> List[str]:
        """All triggered signal names across all components."""
        signals = list(self.behavior_signals)
        if self.enum_detected:
            signals.append("sequential_enumeration")
        if not self.param_valid:
            signals.append("param_violation")
        return signals

    def to_module_risk_dict(self) -> Dict:
        """Export in ModuleRisk-compatible format for fusion gateway."""
        return {
            "module": "output_agency",
            "risk_score": round(self.risk_score, 4),
            "confidence": round(self.confidence, 4),
            "decision": self.decision,
            "evidence": self.evidence,
            "latency_ms": self.latency_ms,
        }

    def to_detailed_dict(self) -> Dict:
        """Export with full component breakdown for evaluation/logging."""
        return {
            "user_id": self.user_id,
            "tool": self.tool,
            "behavior_risk_score": round(self.risk_score, 4),
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "signals_triggered": self.signals_triggered,
            "component_scores": {
                "behavior": round(self.behavior_score, 4),
                "enum": round(self.enum_score, 4),
                "param": round(self.param_score, 4),
            },
            "behavior_risk_level": self.behavior_risk_level,
            "enum_detected": self.enum_detected,
            "param_valid": self.param_valid,
            "latency_ms": self.latency_ms,
        }


# ---------------------------------------------------------------------------
# Risk Model
# ---------------------------------------------------------------------------
class BehaviorRiskModel:
    """
    Unified behavioral risk model for agency defense.

    Combines three signal sources with weights:
    - behavior_monitor: 0.40 (burst, diversity, repeat, failed auth, lateral)
    - anti_enum_guard:  0.35 (sequential ID probing)
    - param_validation: 0.25 (malformed/malicious parameters)

    Final risk = max(weighted_sum, highest_component * 0.85)
    → Ensures a single critical signal can trigger block alone.
    """

    def __init__(
        self,
        monitor: Optional[BehaviorMonitor] = None,
        enum_guard: Optional[AntiEnumGuard] = None,
        param_validator: Optional[ParameterValidator] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.monitor = monitor or BehaviorMonitor()
        self.enum_guard = enum_guard or AntiEnumGuard()
        self.param_validator = param_validator or ParameterValidator()
        self.weights = weights or {
            "behavior": 0.40,
            "enum": 0.35,
            "param": 0.25,
        }

    def assess(
        self,
        user_id: str,
        tool: str,
        resource_id: str = "",
        args: Optional[Dict[str, Any]] = None,
        resource_type: Optional[str] = None,
        was_authorized: bool = True,
    ) -> BehaviorRiskResult:
        """
        Run full behavioral risk assessment.

        Args:
            user_id:        User making the call.
            tool:           Tool being called.
            resource_id:    Resource being accessed.
            args:           Tool arguments for parameter validation.
            resource_type:  Type of resource (for lateral movement tracking).
            was_authorized: Whether authorization passed.

        Returns:
            BehaviorRiskResult with combined risk score and decision.
        """
        t0 = time.time()
        args = args or {}
        evidence = []

        # --- Signal 1: Behavior Monitor ---
        behavior = self.monitor.record(
            user_id, tool, resource_id,
            resource_type=resource_type,
            was_authorized=was_authorized,
        )
        behavior_score = behavior.risk_score
        if behavior.risk_level != "low":
            evidence.extend(behavior.evidence)

        # --- Signal 2: Anti-Enumeration ---
        enum_score = 0.0
        enum_detected = False
        if resource_id:
            enum_result = self.enum_guard.check(user_id, resource_id)
            enum_score = enum_result.risk_score
            enum_detected = enum_result.is_enumeration
            if enum_detected:
                evidence.extend(enum_result.evidence)

        # --- Signal 3: Parameter Validation ---
        param_result = self.param_validator.validate(tool, args)
        param_score = param_result.risk_contribution
        param_valid = param_result.is_valid
        if not param_valid:
            evidence.extend(param_result.violations)

        # --- Combined Risk ---
        w = self.weights
        weighted = (
            w["behavior"] * behavior_score
            + w["enum"] * enum_score
            + w["param"] * param_score
        )

        # Max rule: single critical signal can override
        highest = max(behavior_score, enum_score, param_score)
        risk_score = round(min(max(weighted, highest * 0.85), 1.0), 4)

        decision = _decide(risk_score)

        # Confidence scales with observation count
        if behavior.total_calls >= 5:
            confidence = 0.90
        elif behavior.total_calls >= 3:
            confidence = 0.80
        else:
            confidence = 0.65

        if not evidence:
            evidence.append(
                f"Normal behavior: {behavior.total_calls} calls, no anomalies"
            )

        evidence.append(
            f"Components: behavior={behavior_score:.3f}, "
            f"enum={enum_score:.3f}, param={param_score:.3f}"
        )

        latency_ms = int((time.time() - t0) * 1000)

        return BehaviorRiskResult(
            user_id=user_id,
            tool=tool,
            risk_score=risk_score,
            decision=decision,
            confidence=confidence,
            evidence=evidence,
            latency_ms=latency_ms,
            behavior_score=round(behavior_score, 4),
            enum_score=round(enum_score, 4),
            param_score=round(param_score, 4),
            behavior_risk_level=behavior.risk_level,
            behavior_signals=behavior.signals,
            enum_detected=enum_detected,
            param_valid=param_valid,
        )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Setup
    monitor = BehaviorMonitor(
        window_seconds=60, burst_threshold=8,
        resource_diversity_threshold=6, failed_auth_threshold=3,
    )
    enum_guard = AntiEnumGuard(window_seconds=60, seq_threshold=3)
    validator = ParameterValidator()
    validator.register_tool_schema("get_order", {
        "resource_id": {
            "type": "str", "required": True,
            "max_length": 50, "pattern": r"^[A-Z]+-\d+$",
        },
    })

    model = BehaviorRiskModel(monitor, enum_guard, validator)

    print(f"{'='*65}\n  BEHAVIOR RISK MODEL DEMO\n{'='*65}")

    # Scenario 1: Normal usage
    print(f"\n  [Scenario 1] Normal user — 3 legit calls:")
    model.monitor.reset_all()
    model.enum_guard.reset_all()
    for rid in ["ORD-001", "ORD-003", "TKT-101"]:
        r = model.assess("alice", "get_order", rid,
                         {"resource_id": rid}, resource_type="order")
    print(f"    Risk: {r.risk_score:.3f} | Decision: {r.decision} "
          f"| Level: {r.behavior_risk_level}")

    # Scenario 2: Sequential enumeration
    print(f"\n  [Scenario 2] Sequential ID probing:")
    model.monitor.reset_all()
    model.enum_guard.reset_all()
    for i in range(1001, 1006):
        rid = f"ORD-{i}"
        r = model.assess("attacker", "get_order", rid,
                         {"resource_id": rid}, resource_type="order")
        if r.enum_detected or i == 1001:
            print(f"    {rid}: risk={r.risk_score:.3f} | decision={r.decision} "
                  f"| enum={r.enum_detected}")

    # Scenario 3: SQL injection in params
    print(f"\n  [Scenario 3] SQL injection in resource_id:")
    model.monitor.reset_all()
    model.enum_guard.reset_all()
    r = model.assess("attacker", "get_order", "ORD-001",
                     {"resource_id": "ORD-001'; DROP TABLE;--"}, resource_type="order")
    print(f"    Risk: {r.risk_score:.3f} | Decision: {r.decision} "
          f"| Valid params: {r.param_valid}")

    # Scenario 4: Burst + failed auth + lateral
    print(f"\n  [Scenario 4] Combined: burst + failures + lateral:")
    model.monitor.reset_all()
    model.enum_guard.reset_all()
    types = ["order", "ticket", "identity", "config"]
    for i in range(12):
        rid = f"RES-{i:04d}"
        rt = types[i % len(types)]
        r = model.assess("attacker", "get_order", rid,
                         {"resource_id": rid}, resource_type=rt,
                         was_authorized=(i < 2))
    print(f"    Risk: {r.risk_score:.3f} | Decision: {r.decision}")
    print(f"    Behavior: {r.behavior_risk_level} | Enum: {r.enum_detected} "
          f"| Params: {r.param_valid}")
    print(f"    Signals: {r.behavior_signals}")

    print(f"\n{'='*65}")
