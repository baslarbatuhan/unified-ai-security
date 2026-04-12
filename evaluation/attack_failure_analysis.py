"""
evaluation/attack_failure_analysis.py
=========================================
Attack Success/Failure Analysis.

Parses gateway_attack_results.csv, identifies which attacks escaped
detection, groups by attack type and module, and generates a
markdown report.

Output:
    reports/attack_failure_analysis.md

Usage:
    python evaluation/attack_failure_analysis.py
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_RUNS_DIR = _PROJECT_ROOT / "runs"
_REPORTS_DIR = _PROJECT_ROOT / "reports"


def load_results() -> list:
    """Load attack results CSV."""
    csv_path = _RUNS_DIR / "gateway_attack_results.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run attack suite first.")
        sys.exit(1)
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def analyze(results: list) -> str:
    """Generate markdown analysis report."""
    total = len(results)
    blocked = [r for r in results if r["decision"] in ("block", "flag")]
    escaped = [r for r in results if r["decision"] not in ("block", "flag")]

    # Group escaped by module and attack type
    escaped_by_module = defaultdict(list)
    escaped_by_type = defaultdict(list)
    for r in escaped:
        escaped_by_module[r["module_targeted"]].append(r)
        escaped_by_type[r["attack_type"]].append(r)

    # Group all by module for detection rate
    by_module = defaultdict(lambda: {"total": 0, "blocked": 0})
    for r in results:
        mod = r["module_targeted"]
        by_module[mod]["total"] += 1
        if r["decision"] in ("block", "flag"):
            by_module[mod]["blocked"] += 1

    lines = []
    lines.append("# Attack Failure Analysis Report\n")
    lines.append(f"**Total attacks:** {total}  ")
    lines.append(f"**Blocked/Flagged:** {len(blocked)} ({len(blocked)/total*100:.1f}%)  ")
    lines.append(f"**Escaped (allow/sanitize):** {len(escaped)} ({len(escaped)/total*100:.1f}%)\n")

    lines.append("## Per-Module Detection Rate\n")
    lines.append("| Module | Total | Blocked | Rate |")
    lines.append("|--------|-------|---------|------|")
    for mod, stats in sorted(by_module.items()):
        rate = stats["blocked"] / stats["total"] * 100 if stats["total"] > 0 else 0
        lines.append(f"| {mod} | {stats['total']} | {stats['blocked']} | {rate:.1f}% |")

    lines.append("\n## Escaped Attacks by Module\n")
    if not escaped:
        lines.append("No attacks escaped detection.\n")
    else:
        for mod, attacks in sorted(escaped_by_module.items()):
            lines.append(f"### {mod} ({len(attacks)} escaped)\n")
            for a in attacks:
                lines.append(f"- **{a['attack_id']}** ({a['attack_type']})")
                lines.append(f"  - Decision: {a['decision']} | Fused: {a['fused_risk_score']}")
                lines.append(f"  - Scores: prompt={a['module_prompt_score']}, rag={a['module_rag_score']}, agency={a['module_agency_score']}")
                lines.append(f"  - Override: {a['override_triggered']}")
            lines.append("")

    lines.append("## Escaped Attacks by Type\n")
    if not escaped:
        lines.append("No attacks escaped detection.\n")
    else:
        lines.append("| Attack Type | Count | Avg Fused Score |")
        lines.append("|-------------|-------|-----------------|")
        for atype, attacks in sorted(escaped_by_type.items()):
            avg_fused = sum(float(a["fused_risk_score"]) for a in attacks) / len(attacks)
            lines.append(f"| {atype} | {len(attacks)} | {avg_fused:.4f} |")

    lines.append("\n## Weakest Links\n")
    # Find modules with lowest detection rate
    for mod, stats in sorted(by_module.items(), key=lambda x: x[1]["blocked"]/max(1, x[1]["total"])):
        rate = stats["blocked"] / stats["total"] * 100 if stats["total"] > 0 else 0
        if rate < 80:
            lines.append(f"- **{mod}**: {rate:.1f}% detection rate - needs improvement")

    return "\n".join(lines)


def run_analysis():
    """Run analysis and save report."""
    print(f"\n{'='*65}")
    print(f"  ATTACK FAILURE ANALYSIS")
    print(f"{'='*65}")

    results = load_results()
    report = analyze(results)

    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _REPORTS_DIR / "attack_failure_analysis.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  [Saved] {report_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    run_analysis()
