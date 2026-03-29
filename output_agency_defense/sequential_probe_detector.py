"""
output_agency_defense/sequential_probe_detector.py
====================================================
Detects sequential ID probing / brute-force enumeration attempts.

Purpose:
    - Track resource_id attempts per user over time
    - Compute delta between consecutive ID attempts
    - Flag sequential patterns (e.g., 1001 → 1002 → 1003 → 1004)
    - Raise risk score when probing threshold is exceeded

Detection Logic:
    - Extract numeric parts from resource IDs
    - Compute deltas between consecutive attempts
    - If delta == 1 for N consecutive attempts → enumeration detected
    - Track unique_id_attempts_per_minute

Metrics Tracked:
    - unique_id_attempts_per_minute
    - sequential_id_delta

Usage:
    detector = SequentialProbeDetector(window_seconds=60, seq_threshold=3)
    result = detector.record_attempt("user_bob", "ORD-1001")
    result = detector.record_attempt("user_bob", "ORD-1002")
    result = detector.record_attempt("user_bob", "ORD-1003")
    # result.is_probing == True, result.sequential_count == 3
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Probe result
# ---------------------------------------------------------------------------
@dataclass
class ProbeResult:
    """Result of a single probe detection check."""
    user_id: str
    resource_id: str
    is_probing: bool = False
    sequential_count: int = 0
    unique_attempts_in_window: int = 0
    max_delta: int = 0
    risk_contribution: float = 0.0
    evidence: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# User attempt record
# ---------------------------------------------------------------------------
@dataclass
class UserAttemptRecord:
    """Tracks a user's resource access attempts over time."""
    attempts: List[Tuple[float, str, Optional[int]]] = field(default_factory=list)
    # (timestamp, resource_id, numeric_part_or_None)


# ---------------------------------------------------------------------------
# Sequential Probe Detector
# ---------------------------------------------------------------------------
class SequentialProbeDetector:
    """
    Detects sequential ID enumeration attempts.

    Tracks per-user resource_id access patterns and flags:
    1. Sequential ID deltas (ORD-1001 → ORD-1002 → ORD-1003)
    2. High unique ID attempt rate within a time window
    3. Numeric brute-force patterns

    Example:
        1001 → 1002 → 1003 → 1004 gibi denemeler tespit edildiğinde
        risk skoru yükseltilir.
    """

    def __init__(
        self,
        window_seconds: int = 60,
        seq_threshold: int = 3,
        rate_threshold: int = 10,
    ):
        """
        Args:
            window_seconds: Time window for rate tracking (default 60s)
            seq_threshold:  Consecutive sequential IDs to trigger alert (default 3)
            rate_threshold: Max unique IDs per window before flagging (default 10)
        """
        self.window_seconds = window_seconds
        self.seq_threshold = seq_threshold
        self.rate_threshold = rate_threshold
        self._user_records: Dict[str, UserAttemptRecord] = defaultdict(UserAttemptRecord)

    @staticmethod
    def _extract_numeric(resource_id: str) -> Optional[int]:
        """Extract numeric suffix from a resource ID."""
        match = re.search(r'(\d+)$', resource_id)
        if match:
            return int(match.group(1))
        # Try extracting any number
        numbers = re.findall(r'\d+', resource_id)
        if numbers:
            return int(numbers[-1])
        return None

    def _prune_old_attempts(self, record: UserAttemptRecord, now: float):
        """Remove attempts outside the time window."""
        cutoff = now - self.window_seconds
        record.attempts = [a for a in record.attempts if a[0] >= cutoff]

    def record_attempt(self, user_id: str, resource_id: str) -> ProbeResult:
        """
        Record a resource access attempt and check for probing.

        Args:
            user_id:     The user making the attempt
            resource_id: The resource ID being accessed

        Returns:
            ProbeResult with detection signals.
        """
        now = time.time()
        numeric = self._extract_numeric(resource_id)
        record = self._user_records[user_id]

        # Add new attempt
        record.attempts.append((now, resource_id, numeric))

        # Prune old attempts
        self._prune_old_attempts(record, now)

        # --- Signal 1: Unique IDs in window ---
        unique_ids = set(a[1] for a in record.attempts)
        unique_count = len(unique_ids)

        # --- Signal 2: Sequential delta analysis ---
        numeric_attempts = [(ts, rid, num) for ts, rid, num in record.attempts if num is not None]
        numeric_attempts.sort(key=lambda x: x[0])  # sort by time

        sequential_count = 0
        max_consecutive = 0
        current_streak = 0

        if len(numeric_attempts) >= 2:
            for i in range(1, len(numeric_attempts)):
                prev_num = numeric_attempts[i-1][2]
                curr_num = numeric_attempts[i][2]
                delta = abs(curr_num - prev_num)

                if delta == 1:
                    current_streak += 1
                    max_consecutive = max(max_consecutive, current_streak)
                else:
                    current_streak = 0

            sequential_count = max_consecutive + 1  # +1 for the first ID in streak

        # --- Determine if probing ---
        evidence = []
        is_probing = False
        risk_contribution = 0.0

        # Check sequential threshold
        if sequential_count >= self.seq_threshold:
            is_probing = True
            risk_contribution = min(0.3 + (sequential_count - self.seq_threshold) * 0.15, 1.0)
            evidence.append(
                f"Sequential probe detected: {sequential_count} consecutive IDs "
                f"(threshold: {self.seq_threshold})"
            )

        # Check rate threshold
        if unique_count >= self.rate_threshold:
            is_probing = True
            rate_risk = min(0.4 + (unique_count - self.rate_threshold) * 0.1, 1.0)
            risk_contribution = max(risk_contribution, rate_risk)
            evidence.append(
                f"High ID attempt rate: {unique_count} unique IDs in {self.window_seconds}s "
                f"(threshold: {self.rate_threshold})"
            )

        if not is_probing:
            evidence.append(f"No probing detected: {unique_count} unique IDs, {sequential_count} sequential")

        return ProbeResult(
            user_id=user_id,
            resource_id=resource_id,
            is_probing=is_probing,
            sequential_count=sequential_count,
            unique_attempts_in_window=unique_count,
            max_delta=max_consecutive,
            risk_contribution=round(risk_contribution, 4),
            evidence=evidence,
        )

    def get_user_stats(self, user_id: str) -> Dict:
        """Get current stats for a user."""
        record = self._user_records.get(user_id)
        if not record:
            return {"user_id": user_id, "attempts": 0}
        self._prune_old_attempts(record, time.time())
        return {
            "user_id": user_id,
            "attempts_in_window": len(record.attempts),
            "unique_ids": len(set(a[1] for a in record.attempts)),
        }

    def reset_user(self, user_id: str):
        """Clear tracking data for a user."""
        if user_id in self._user_records:
            del self._user_records[user_id]

    def reset_all(self):
        """Clear all tracking data."""
        self._user_records.clear()


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    detector = SequentialProbeDetector(window_seconds=60, seq_threshold=3, rate_threshold=10)

    print(f"{'='*60}")
    print(f"  SEQUENTIAL PROBE DETECTOR DEMO")
    print(f"{'='*60}")

    # Scenario 1: Normal user accessing own resources
    print(f"\n  [Scenario 1] Normal access pattern:")
    for rid in ["ORD-001", "ORD-003", "ORD-007"]:
        r = detector.record_attempt("user_alice", rid)
        print(f"    {rid}: probing={r.is_probing} | sequential={r.sequential_count} | unique={r.unique_attempts_in_window}")

    # Scenario 2: Sequential probing attack
    print(f"\n  [Scenario 2] Sequential ID probing:")
    detector.reset_all()
    for i in range(1001, 1008):
        rid = f"ORD-{i}"
        r = detector.record_attempt("attacker_bob", rid)
        status = "PROBING" if r.is_probing else "ok"
        print(f"    {rid}: [{status}] sequential={r.sequential_count} | risk={r.risk_contribution:.2f}")
        if r.evidence and r.is_probing:
            for e in r.evidence:
                print(f"      → {e}")

    # Scenario 3: Random ID guessing (high rate)
    print(f"\n  [Scenario 3] Random ID guessing (high rate):")
    detector.reset_all()
    import random
    random.seed(42)
    for i in range(12):
        rid = f"ORD-{random.randint(1, 9999):04d}"
        r = detector.record_attempt("attacker_eve", rid)
        status = "RATE_LIMIT" if r.is_probing else "ok"
        print(f"    {rid}: [{status}] unique={r.unique_attempts_in_window} | risk={r.risk_contribution:.2f}")

    print(f"\n{'='*60}")
