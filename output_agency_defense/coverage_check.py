"""
output_agency_defense/coverage_check.py
=========================================
Tool coverage validation.

Purpose:
    - Verify ALL tools are registered with secure_tool_wrapper
    - Ensure no tool can bypass the security layer
    - Report coverage gaps

Usage:
    python output_agency_defense/coverage_check.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class CoverageReport:
    """Result of tool coverage validation."""
    total_known_tools: int
    total_wrapped_tools: int
    covered: List[str] = field(default_factory=list)
    uncovered: List[str] = field(default_factory=list)
    coverage_ratio: float = 0.0
    is_fully_covered: bool = False
    evidence: List[str] = field(default_factory=list)


class ToolCoverageChecker:
    """
    Validates that all known tools are registered with the secure_tool_wrapper.

    Any tool not going through the wrapper is a security gap —
    it bypasses authorization, audit logging, and anti-enumeration checks.
    """

    def __init__(self):
        self._known_tools: Set[str] = set()
        self._known_tool_meta: Dict[str, Dict] = {}

    def register_known_tool(self, name: str, description: str = "", category: str = ""):
        """Register a tool that the system knows about (from tool definitions)."""
        self._known_tools.add(name)
        self._known_tool_meta[name] = {"description": description, "category": category}

    def register_known_tools(self, tools: List[Dict]):
        """Bulk register known tools."""
        for t in tools:
            self.register_known_tool(t.get("name", ""), t.get("description", ""), t.get("category", ""))

    def check_coverage(self, wrapped_tools: List[str]) -> CoverageReport:
        """
        Check which known tools are covered by the wrapper.

        Args:
            wrapped_tools: List of tool names registered in secure_tool_wrapper

        Returns:
            CoverageReport with coverage details.
        """
        wrapped_set = set(wrapped_tools)
        covered = sorted(self._known_tools & wrapped_set)
        uncovered = sorted(self._known_tools - wrapped_set)

        total_known = len(self._known_tools)
        total_wrapped = len(covered)
        ratio = total_wrapped / total_known if total_known > 0 else 1.0
        is_full = len(uncovered) == 0

        evidence = []
        if is_full:
            evidence.append(f"All {total_known} tools are covered by secure_tool_wrapper")
        else:
            evidence.append(f"COVERAGE GAP: {len(uncovered)} tools bypass security layer")
            for tool in uncovered:
                meta = self._known_tool_meta.get(tool, {})
                evidence.append(f"  Uncovered: '{tool}' — {meta.get('description', 'no description')}")

        # Check for wrapper-only tools (wrapped but not in known — possible stale registration)
        extra = sorted(wrapped_set - self._known_tools)
        if extra:
            evidence.append(f"Note: {len(extra)} wrapped tools not in known tools list: {extra}")

        return CoverageReport(
            total_known_tools=total_known,
            total_wrapped_tools=total_wrapped,
            covered=covered,
            uncovered=uncovered,
            coverage_ratio=round(ratio, 4),
            is_fully_covered=is_full,
            evidence=evidence,
        )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    checker = ToolCoverageChecker()

    # Define all known tools in the system
    known_tools = [
        {"name": "get_order", "description": "Fetch order details", "category": "read"},
        {"name": "cancel_order", "description": "Cancel an order", "category": "write"},
        {"name": "get_ticket", "description": "Fetch ticket info", "category": "read"},
        {"name": "update_ticket", "description": "Update ticket status", "category": "write"},
        {"name": "system_status", "description": "Check system health", "category": "admin"},
        {"name": "delete_user", "description": "Delete a user account", "category": "admin"},
    ]
    checker.register_known_tools(known_tools)

    print(f"{'='*55}")
    print(f"  TOOL COVERAGE CHECK")
    print(f"{'='*55}")

    # Scenario 1: Full coverage
    print(f"\n  [Scenario 1] All tools wrapped:")
    wrapped = ["get_order", "cancel_order", "get_ticket", "update_ticket", "system_status", "delete_user"]
    report = checker.check_coverage(wrapped)
    print(f"    Coverage: {report.coverage_ratio:.0%} | Full: {report.is_fully_covered}")
    for e in report.evidence:
        print(f"    {e}")

    # Scenario 2: Missing tools
    print(f"\n  [Scenario 2] Some tools missing from wrapper:")
    wrapped_partial = ["get_order", "cancel_order", "get_ticket", "system_status"]
    report = checker.check_coverage(wrapped_partial)
    print(f"    Coverage: {report.coverage_ratio:.0%} | Full: {report.is_fully_covered}")
    for e in report.evidence:
        print(f"    {e}")

    # Scenario 3: No tools wrapped
    print(f"\n  [Scenario 3] No tools wrapped (security bypass):")
    report = checker.check_coverage([])
    print(f"    Coverage: {report.coverage_ratio:.0%} | Full: {report.is_fully_covered}")
    for e in report.evidence:
        print(f"    {e}")

    print(f"\n{'='*55}")
