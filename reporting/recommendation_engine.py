"""reporting/recommendation_engine.py
Heuristic recommendations derived from a metrics snapshot.

Each rule is a small declarative entry — predicate over the snapshot, severity,
human-readable text, and optional links to which artefacts to inspect. We
intentionally keep this as a curated list rather than a YAML config: the rules
encode product knowledge ("if bypass rate > 5%, look at fusion thresholds") and
we want diff-able review when they change.

The engine returns a sorted list of Recommendations (highest severity first),
which `report_generator` renders into a markdown section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


Severity = str  # "info" | "warn" | "critical"


@dataclass
class Recommendation:
    rule_id: str
    severity: Severity
    title: str
    detail: str
    artefacts: List[str] = field(default_factory=list)
    metric_values: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "artefacts": list(self.artefacts),
            "metric_values": dict(self.metric_values),
        }


# ---------------------------------------------------------------------------
# Rule definitions
# Each rule = (id, severity, predicate, title, detail-template, artefacts)
# Detail template is a `.format`-style string with snapshot keys as kwargs.
# Predicate receives the snapshot dict and returns a bool.
# ---------------------------------------------------------------------------
def _gt(metric: str, threshold: float) -> Callable[[Dict[str, float]], bool]:
    return lambda s: float(s.get(metric, 0.0) or 0.0) > threshold


def _lt(metric: str, threshold: float) -> Callable[[Dict[str, float]], bool]:
    return lambda s: float(s.get(metric, 0.0) or 0.0) < threshold


_RULES: List[Dict[str, Any]] = [
    {
        "id": "tune_fusion_thresholds",
        "severity": "critical",
        "predicate": _gt("attack_bypass_rate", 0.05),
        "title": "Fusion thresholds let attacks through",
        "detail": (
            "Bypass proxy is {attack_bypass_rate:.1%} — at least one module "
            "scored ≥ 0.60 but the fused decision was `allow`. Lower the "
            "elevated/critical thresholds in `configs/fusion_weights.yaml` "
            "or raise the override multiplier so high-confidence single-module "
            "signals can dominate."
        ),
        "artefacts": ["runs/perf_results.csv", "configs/fusion_weights.yaml"],
        "metrics": ["attack_bypass_rate"],
    },
    {
        "id": "investigate_p95_latency",
        "severity": "warn",
        "predicate": _gt("p95_latency_ms", 2000.0),
        "title": "p95 latency over budget",
        "detail": (
            "p95 latency is {p95_latency_ms:.0f} ms (>2000 ms budget). Most "
            "common cause: cold-start of the LLM judge on the first request. "
            "Check `runs/judge_determinism_metrics.csv` and consider warming "
            "the judge at startup or routing chunks via the chunk_router fast "
            "path more aggressively."
        ),
        "artefacts": ["reports/judge_determinism.md", "rag_guard/chunk_router.py"],
        "metrics": ["p95_latency_ms"],
    },
    {
        "id": "high_error_rate",
        "severity": "critical",
        "predicate": _gt("error_rate", 0.02),
        "title": "Error rate exceeds 2%",
        "detail": (
            "Module errors are {error_rate:.2%} of requests. Inspect circuit "
            "breakers — when LLM judge or rag_guard raises repeatedly, the "
            "pipeline degrades silently. See operations dashboard."
        ),
        "artefacts": ["dashboard/pages/4_live_monitor.py", "logs/system_telemetry.jsonl"],
        "metrics": ["error_rate"],
    },
    {
        "id": "low_block_high_traffic",
        "severity": "warn",
        "predicate": lambda s: (
            float(s.get("total_requests", 0) or 0) >= 50
            and float(s.get("block_rate", 0.0) or 0.0) < 0.01
        ),
        "title": "No blocks despite meaningful traffic",
        "detail": (
            "{total_requests:.0f} requests evaluated and block rate is "
            "{block_rate:.2%}. Either traffic is genuinely benign or the "
            "fusion thresholds are too lenient. Cross-check against the "
            "regression set: a healthy gateway should block 10–25% of "
            "`datasets/prompt_regression_set.json` attack cases."
        ),
        "artefacts": ["datasets/prompt_regression_set.json"],
        "metrics": ["total_requests", "block_rate"],
    },
    {
        "id": "rag_module_slow",
        "severity": "warn",
        "predicate": _gt("module_rag_guard_avg_latency_ms", 1500.0),
        "title": "rag_guard latency dominates",
        "detail": (
            "rag_guard average is {module_rag_guard_avg_latency_ms:.0f} ms. "
            "The chunk_router can skip the LLM judge for low-embedding-score "
            "chunks; verify the embedding gate is enabled in "
            "`configs/secure_balanced.yaml` and tune `embedding_gate_threshold` "
            "via `evaluation/chunking_sweep.py`."
        ),
        "artefacts": ["evaluation/chunking_sweep.py", "configs/secure_balanced.yaml"],
        "metrics": ["module_rag_guard_avg_latency_ms"],
    },
    {
        "id": "judge_breaker_open",
        "severity": "critical",
        "predicate": lambda s: float(s.get("breaker_llm_judge_open", 0.0) or 0.0) >= 1.0,
        "title": "LLM judge circuit breaker is open",
        "detail": (
            "The judge dependency tripped its breaker — every poisoned-doc "
            "verdict for the cooldown window relies on embedding similarity "
            "alone. This is the lower-precision fallback; resolve the upstream "
            "(usually Ollama) before claiming the documented poison-detection "
            "F1."
        ),
        "artefacts": ["dashboard/pages/4_live_monitor.py"],
        "metrics": ["breaker_llm_judge_open"],
    },
    {
        "id": "agency_guard_inactive",
        "severity": "info",
        "predicate": lambda s: (
            float(s.get("module_output_agency_avg_latency_ms", 0.0) or 0.0) == 0.0
            and float(s.get("total_requests", 0) or 0) >= 10
        ),
        "title": "agency guard never invoked",
        "detail": (
            "No tool calls observed across {total_requests:.0f} requests, so "
            "output_agency contributed zero signal. If your evaluation suite "
            "includes tool-using attacks, verify the agency module is enabled "
            "and a `tool_request` is reaching the gateway."
        ),
        "artefacts": ["datasets/agency_demo_cases.json", "output_agency_defense/"],
        "metrics": ["module_output_agency_avg_latency_ms", "total_requests"],
    },
]


_SEVERITY_RANK = {"critical": 2, "warn": 1, "info": 0}


def derive_recommendations(snapshot: Dict[str, float]) -> List[Recommendation]:
    """Run every rule against the snapshot and return a sorted list."""
    fired: List[Recommendation] = []
    for rule in _RULES:
        try:
            if not rule["predicate"](snapshot):
                continue
        except Exception:
            # Bad predicate must not stop other rules.
            continue
        try:
            detail = rule["detail"].format(**{k: snapshot.get(k, 0.0) for k in rule.get("metrics", [])})
        except Exception:
            detail = rule["detail"]
        fired.append(Recommendation(
            rule_id=rule["id"],
            severity=rule["severity"],
            title=rule["title"],
            detail=detail,
            artefacts=list(rule.get("artefacts", [])),
            metric_values={k: float(snapshot.get(k, 0.0) or 0.0) for k in rule.get("metrics", [])},
        ))
    fired.sort(key=lambda r: -_SEVERITY_RANK.get(r.severity, 0))
    return fired


def render_recommendations(recs: List[Recommendation]) -> str:
    """Markdown section. Empty list → 'no actions' note."""
    lines: List[str] = ["## Recommendations", ""]
    if not recs:
        lines.append("_No actionable items — system metrics are within targets._")
        lines.append("")
        return "\n".join(lines)

    for r in recs:
        badge = {"critical": "🔴", "warn": "🟡", "info": "🔵"}.get(r.severity, "•")
        lines.append(f"### {badge} `{r.rule_id}` — {r.title}")
        lines.append("")
        lines.append(r.detail)
        if r.artefacts:
            arts = ", ".join(f"`{a}`" for a in r.artefacts)
            lines.append("")
            lines.append(f"**See also:** {arts}")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "Recommendation",
    "derive_recommendations",
    "render_recommendations",
]
