"""
output_agency_defense/behavior_monitor.py
============================================
Behavioral risk monitoring for tool calls.

Purpose:
    Track user behavior patterns over time and detect:
    1. Burst calls         — too many requests in short window
    2. Resource diversity   — many different resource_ids (reconnaissance)
    3. Tool repetition      — same tool called excessively (brute-force)
    4. Failed auth          — multiple denied requests (strongest signal)
    5. Mixed resource access — different resource types (lateral movement)

    Unlike anti_enum_guard (which detects sequential IDs),
    this monitors OVERALL behavior patterns.

Integration:
    Called by behavior_risk_model.py which combines this with
    anti_enum_guard and parameter_validation signals.

Usage:
    monitor = BehaviorMonitor(window_seconds=60)
    result = monitor.record("user_bob", "get_order", "ORD-001",
                            resource_type="order", was_authorized=False)
    # result.risk_level == "medium", result.signals == ["failed_auth"]
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


RiskLevel = Literal["low", "medium", "high", "critical"]


# ---------------------------------------------------------------------------
# Event + Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class BehaviorEvent:
    """Single recorded event."""
    timestamp: float
    user_id: str
    tool: str
    resource_id: str
    resource_type: Optional[str]
    was_authorized: bool


@dataclass
class BehaviorAssessment:
    """Result of behavior analysis for a single user."""
    user_id: str
    risk_level: RiskLevel = "low"
    risk_score: float = 0.0
    signals: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)

    # Aggregate metrics
    total_calls: int = 0
    calls_per_minute: float = 0.0
    unique_resources: int = 0
    unique_tools: int = 0
    unique_resource_types: int = 0
    failed_auth_count: int = 0
    most_called_tool: str = ""
    most_called_count: int = 0

    # Per-signal scores (0.0 - 1.0) for granular analysis / CSV export
    burst_score: float = 0.0
    diversity_score: float = 0.0
    repetition_score: float = 0.0
    failure_score: float = 0.0
    lateral_score: float = 0.0


# ---------------------------------------------------------------------------
# Behavior Monitor
# ---------------------------------------------------------------------------
class BehaviorMonitor:
    """
    Monitors user behavior patterns across tool calls.

    Tracks per-user within a sliding time window:
    - Request rate (burst detection)
    - Resource diversity (unique resource_ids)
    - Tool repetition (same tool called many times)
    - Authorization failures (blocked requests)
    - Mixed resource access (lateral movement across resource types)

    Risk levels:
    - low:      0 signals active
    - medium:   1 signal
    - high:     2 signals
    - critical: 3+ signals (or failed_auth alone with high count)
    """

    # Weights for weighted risk calculation
    WEIGHTS = {
        "burst": 0.20,
        "diversity": 0.20,
        "repetition": 0.15,
        "failure": 0.35,
        "lateral": 0.10,
    }

    def __init__(
        self,
        window_seconds: int = 60,
        burst_threshold: int = 15,
        resource_diversity_threshold: int = 8,
        tool_repeat_threshold: int = 10,
        failed_auth_threshold: int = 3,
    ):
        self.window_seconds = window_seconds
        self.burst_threshold = burst_threshold
        self.resource_diversity_threshold = resource_diversity_threshold
        self.tool_repeat_threshold = tool_repeat_threshold
        self.failed_auth_threshold = failed_auth_threshold
        self._events: Dict[str, List[BehaviorEvent]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _prune(self, user_id: str, now: float):
        cutoff = now - self.window_seconds
        self._events[user_id] = [
            e for e in self._events[user_id] if e.timestamp >= cutoff
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def record(
        self,
        user_id: str,
        tool: str,
        resource_id: str = "",
        resource_type: Optional[str] = None,
        was_authorized: bool = True,
    ) -> BehaviorAssessment:
        """
        Record a tool call event and assess behavior risk.

        Args:
            user_id:        The user making the call.
            tool:           Tool name being called.
            resource_id:    Resource being accessed (if any).
            resource_type:  Type of resource (order, ticket, etc.).
            was_authorized: Whether the call was authorized.

        Returns:
            BehaviorAssessment with risk level, score, and signals.
        """
        now = time.time()
        event = BehaviorEvent(now, user_id, tool, resource_id,
                              resource_type, was_authorized)
        self._events[user_id].append(event)
        self._prune(user_id, now)
        return self._assess(user_id)

    def assess_risk(self, user_id: str) -> BehaviorAssessment:
        """Assess risk without recording a new event."""
        self._prune(user_id, time.time())
        return self._assess(user_id)

    def _assess(self, user_id: str) -> BehaviorAssessment:
        """Core assessment logic."""
        events = self._events.get(user_id, [])
        if not events:
            return BehaviorAssessment(user_id=user_id)

        total = len(events)
        elapsed = events[-1].timestamp - events[0].timestamp if total > 1 else self.window_seconds
        cpm = (total / max(elapsed, 1)) * 60

        # --- Gather metrics ---
        unique_resources = set(e.resource_id for e in events if e.resource_id)
        unique_res_count = len(unique_resources)

        tool_counts: Dict[str, int] = defaultdict(int)
        for e in events:
            tool_counts[e.tool] += 1
        most_tool = max(tool_counts, key=tool_counts.get) if tool_counts else ""
        most_count = tool_counts.get(most_tool, 0)

        failed = sum(1 for e in events if not e.was_authorized)

        resource_types = set(e.resource_type for e in events
                             if e.resource_type is not None)
        rt_count = len(resource_types)

        # --- Signal 1: Burst ---
        burst_score = 0.0
        if total >= self.burst_threshold:
            burst_score = min(
                (total - self.burst_threshold + 1) / self.burst_threshold, 1.0
            )

        # --- Signal 2: Resource diversity ---
        diversity_score = 0.0
        if unique_res_count >= self.resource_diversity_threshold:
            diversity_score = min(
                (unique_res_count - self.resource_diversity_threshold + 1)
                / self.resource_diversity_threshold, 1.0
            )

        # --- Signal 3: Tool repetition ---
        repetition_score = 0.0
        if most_count >= self.tool_repeat_threshold:
            repetition_score = min(
                (most_count - self.tool_repeat_threshold + 1)
                / self.tool_repeat_threshold, 1.0
            )

        # --- Signal 4: Failed auth ---
        failure_score = 0.0
        if failed >= self.failed_auth_threshold:
            failure_score = min(
                (failed - self.failed_auth_threshold + 1)
                / self.failed_auth_threshold, 1.0
            )

        # --- Signal 5: Lateral movement (mixed resource types) ---
        lateral_score = 0.0
        if rt_count >= 3:
            lateral_score = min((rt_count - 2) / 3.0, 1.0)

        # --- Build signals + evidence ---
        signals = []
        evidence = []

        if burst_score > 0:
            signals.append("burst")
            evidence.append(
                f"Burst: {total} calls in {self.window_seconds}s "
                f"(threshold: {self.burst_threshold})"
            )
        if diversity_score > 0:
            signals.append("resource_diversity")
            evidence.append(
                f"Resource diversity: {unique_res_count} unique IDs "
                f"(threshold: {self.resource_diversity_threshold})"
            )
        if repetition_score > 0:
            signals.append("tool_repetition")
            evidence.append(
                f"Tool repeat: '{most_tool}' called {most_count}x "
                f"(threshold: {self.tool_repeat_threshold})"
            )
        if failure_score > 0:
            signals.append("failed_auth")
            evidence.append(
                f"Failed auth: {failed} denied requests "
                f"(threshold: {self.failed_auth_threshold})"
            )
        if lateral_score > 0:
            signals.append("lateral_movement")
            evidence.append(
                f"Lateral movement: {rt_count} resource types "
                f"({', '.join(sorted(resource_types))})"
            )

        # --- Risk score (weighted sum) ---
        w = self.WEIGHTS
        risk_score = (
            w["burst"] * burst_score
            + w["diversity"] * diversity_score
            + w["repetition"] * repetition_score
            + w["failure"] * failure_score
            + w["lateral"] * lateral_score
        )
        risk_score = round(min(risk_score, 1.0), 4)

        # --- Risk level (based on signal count + boost for auth failures) ---
        sig_count = len(signals)
        if sig_count == 0:
            risk_level: RiskLevel = "low"
        elif sig_count == 1:
            risk_level = "medium"
        elif sig_count == 2:
            risk_level = "high"
        else:
            risk_level = "critical"

        # Auth failures alone can escalate to critical
        if failed >= self.failed_auth_threshold * 2 and risk_level != "critical":
            risk_level = "critical"

        if not signals:
            evidence.append(
                f"Normal: {total} calls, {unique_res_count} resources, "
                f"{failed} failures"
            )

        return BehaviorAssessment(
            user_id=user_id,
            risk_level=risk_level,
            risk_score=risk_score,
            signals=signals,
            evidence=evidence,
            total_calls=total,
            calls_per_minute=round(cpm, 1),
            unique_resources=unique_res_count,
            unique_tools=len(set(e.tool for e in events)),
            unique_resource_types=rt_count,
            failed_auth_count=failed,
            most_called_tool=most_tool,
            most_called_count=most_count,
            burst_score=round(burst_score, 4),
            diversity_score=round(diversity_score, 4),
            repetition_score=round(repetition_score, 4),
            failure_score=round(failure_score, 4),
            lateral_score=round(lateral_score, 4),
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def get_user_summary(self, user_id: str) -> Dict:
        self._prune(user_id, time.time())
        events = self._events.get(user_id, [])
        return {"user_id": user_id, "events_in_window": len(events)}

    def reset_user(self, user_id: str):
        if user_id in self._events:
            del self._events[user_id]

    def reset_all(self):
        self._events.clear()


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------
def run_scenario(
    name: str,
    events: List[Dict],
    monitor: Optional[BehaviorMonitor] = None,
) -> BehaviorAssessment:
    """
    Run a sequence of events and return the final risk assessment.

    Args:
        name:    Scenario label (for display).
        events:  List of dicts with keys matching record() params:
                 user_id, tool, resource_id, resource_type, was_authorized
        monitor: BehaviorMonitor instance (creates fresh one if None).

    Returns:
        Final BehaviorAssessment after all events.
    """
    if monitor is None:
        monitor = BehaviorMonitor()

    result = None
    for evt in events:
        result = monitor.record(
            user_id=evt["user_id"],
            tool=evt["tool"],
            resource_id=evt.get("resource_id", ""),
            resource_type=evt.get("resource_type"),
            was_authorized=evt.get("was_authorized", True),
        )
    return result or BehaviorAssessment(
        user_id=events[0]["user_id"] if events else "unknown"
    )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    monitor = BehaviorMonitor(
        window_seconds=60, burst_threshold=10,
        resource_diversity_threshold=6, failed_auth_threshold=3,
    )

    print(f"{'='*65}\n  BEHAVIOR MONITOR DEMO\n{'='*65}")

    # Scenario 1: Normal user
    print(f"\n  [Scenario 1] Normal user — 3 calls:")
    monitor.reset_all()
    for rid in ["ORD-001", "ORD-003", "TKT-101"]:
        a = monitor.record("alice", "get_order", rid, "order", True)
    print(f"    Risk: {a.risk_level} ({a.risk_score:.3f}) | Signals: {a.signals or 'none'}")

    # Scenario 2: Burst attack
    print(f"\n  [Scenario 2] Burst attack — 12 rapid calls:")
    monitor.reset_all()
    for i in range(12):
        a = monitor.record("attacker", "get_order", f"ORD-{i:03d}", "order", True)
    print(f"    Risk: {a.risk_level} ({a.risk_score:.3f}) | Signals: {a.signals}")

    # Scenario 3: Failed auth pattern
    print(f"\n  [Scenario 3] Failed auth — 5 denied requests:")
    monitor.reset_all()
    for i in range(5):
        a = monitor.record("attacker", "get_order", f"ORD-{100+i}", "order", False)
    print(f"    Risk: {a.risk_level} ({a.risk_score:.3f}) | Failed: {a.failed_auth_count}")

    # Scenario 4: Lateral movement
    print(f"\n  [Scenario 4] Lateral — 4 resource types:")
    monitor.reset_all()
    types = [("get_order","ORD-1","order"), ("get_ticket","TKT-1","ticket"),
             ("get_identity","ID-1","identity"), ("get_config","CFG-1","config")]
    for tool, rid, rt in types:
        a = monitor.record("recon", tool, rid, rt, True)
    print(f"    Risk: {a.risk_level} ({a.risk_score:.3f}) | Signals: {a.signals}")
    print(f"    Resource types: {a.unique_resource_types}")

    # Scenario 5: Escalation — 10 requests
    print(f"\n  [Scenario 5] Escalation — 10 mixed requests:")
    monitor.reset_all()
    history = []
    for i in range(10):
        a = monitor.record("esc", "get_order", f"ORD-{i:03d}", "order",
                           was_authorized=(i < 2))
        history.append(a.risk_score)
    print(f"    Risk curve: {[round(h, 3) for h in history]}")
    print(f"    Final: {a.risk_level} ({a.risk_score:.3f}) | Signals: {a.signals}")

    print(f"\n{'='*65}")
