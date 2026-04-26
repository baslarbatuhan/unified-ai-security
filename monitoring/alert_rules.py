"""monitoring/alert_rules.py
==============================
Evaluate declarative alert rules from `configs/alert_rules.yaml` against a
metric snapshot. Returns a list of Alert objects — callers decide how to
deliver them (log, webhook, Slack, dashboard badge).

Pipeline
--------
1. `build_snapshot_from_events(events)` — turn telemetry jsonl entries into a
   flat `{metric_name: number}` dict. Dashboard/cron jobs can extend this or
   merge in extra series (circuit breaker state, rate-limit reject count).
2. `evaluate(snapshot, rules=None)` — run every rule's `when:` clauses
   (ANDed) against the snapshot and return fired Alerts.

Design choices
--------------
* Rules are pure data — no Python callbacks. This keeps the config auditable
  from a static file and makes it trivial to unit-test.
* Only scalar operators (`gt/gte/lt/lte/eq`). If a rule wants compound logic
  we express it as two rules with the same severity rather than inventing a
  mini-language.
* `message` supports `str.format(**snapshot)` so alerts can interpolate the
  observed value without evaluating arbitrary expressions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
Severity = str  # "info" | "warn" | "critical"


@dataclass
class Alert:
    rule_id: str
    severity: Severity
    message: str
    metric_values: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
            "metric_values": self.metric_values,
        }


# ---------------------------------------------------------------------------
# Rule loading
# ---------------------------------------------------------------------------
_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "configs" / "alert_rules.yaml"


def load_rules(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = path or _DEFAULT_PATH
    if not Path(p).exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    rules = data.get("rules") or []
    if not isinstance(rules, list):
        raise ValueError(f"alert_rules.yaml: 'rules' must be a list, got {type(rules)}")
    return rules


# ---------------------------------------------------------------------------
# Operator lookup — kept here so typos in YAML surface as a clear KeyError.
# ---------------------------------------------------------------------------
_OPS: Dict[str, Callable[[float, float], bool]] = {
    "gt":  lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "lt":  lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "eq":  lambda a, b: a == b,
}


def _clause_matches(clause: Dict[str, Any], snapshot: Dict[str, float]) -> bool:
    metric = clause["metric"]
    op = clause["op"]
    threshold = float(clause["threshold"])
    if op not in _OPS:
        raise KeyError(f"Unknown op '{op}' in rule clause {clause!r}")
    value = snapshot.get(metric)
    if value is None:
        # Missing metric = rule silently does not fire. Prevents noisy
        # startup alerts before the first evaluation window.
        return False
    try:
        return _OPS[op](float(value), threshold)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
def evaluate(
    snapshot: Dict[str, float],
    rules: Optional[List[Dict[str, Any]]] = None,
) -> List[Alert]:
    """Return every Alert whose clauses all match the snapshot."""
    rules = rules if rules is not None else load_rules()
    fired: List[Alert] = []
    for rule in rules:
        clauses = rule.get("when") or []
        if not clauses:
            continue
        if all(_clause_matches(c, snapshot) for c in clauses):
            msg_tpl = rule.get("message", rule.get("id", "alert"))
            try:
                rendered = msg_tpl.format(**snapshot)
            except (KeyError, IndexError, ValueError):
                rendered = msg_tpl  # fall back to raw template
            # Only surface the metrics referenced by this rule.
            values = {
                c["metric"]: snapshot.get(c["metric"], math.nan)
                for c in clauses
            }
            fired.append(Alert(
                rule_id=str(rule.get("id", "unnamed")),
                severity=str(rule.get("severity", "info")),
                message=rendered,
                metric_values=values,
            ))
    return fired


# ---------------------------------------------------------------------------
# Snapshot builder — reduces a telemetry event stream into metric scalars.
# ---------------------------------------------------------------------------
def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    k = (len(sv) - 1) * (pct / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return sv[lo]
    return sv[lo] + (sv[hi] - sv[lo]) * (k - lo)


def build_snapshot_from_events(events: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Reduce a stream of telemetry jsonl entries into the flat metric dict
    consumed by `evaluate()`. Accepts the raw dicts returned by
    `schemas.telemetry_schema.read_events`.
    """
    fusion_decisions: List[Dict[str, Any]] = []
    module_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for e in events:
        kind = e.get("kind")
        if kind == "fusion_decision":
            fusion_decisions.append(e)
        elif kind == "module_result":
            module_results.append(e)
        elif kind == "error":
            errors.append(e)

    total_requests = len(fusion_decisions)
    snapshot: Dict[str, float] = {
        "total_requests": float(total_requests),
        "block_rate": 0.0,
        "sanitize_rate": 0.0,
        "allow_rate": 0.0,
        "error_rate": 0.0,
        "p95_latency_ms": 0.0,
        "avg_latency_ms": 0.0,
        "attack_bypass_rate": 0.0,
    }

    if total_requests:
        decisions = [d.get("decision") for d in fusion_decisions]
        snapshot["block_rate"] = decisions.count("block") / total_requests
        snapshot["sanitize_rate"] = decisions.count("sanitize") / total_requests
        snapshot["allow_rate"] = decisions.count("allow") / total_requests

        latencies = [float(d.get("latency_ms_total", 0)) for d in fusion_decisions]
        snapshot["avg_latency_ms"] = sum(latencies) / len(latencies)
        snapshot["p95_latency_ms"] = _percentile(latencies, 95.0)

        bypass_candidates = 0
        bypasses = 0
        for d in fusion_decisions:
            # Heuristic: a decision was "blockable" if any module scored ≥ 0.60
            # and the fused decision nonetheless came back as allow.
            module_max = max(
                float(d.get("prompt_score", 0.0)),
                float(d.get("rag_score", 0.0)),
                float(d.get("agency_score", 0.0)),
                float(d.get("output_score", 0.0)),
            )
            if module_max >= 0.60:
                bypass_candidates += 1
                if d.get("decision") == "allow":
                    bypasses += 1
        if bypass_candidates:
            snapshot["attack_bypass_rate"] = bypasses / bypass_candidates

    # Per-module aggregates
    from collections import defaultdict
    per_mod_lat: Dict[str, List[float]] = defaultdict(list)
    per_mod_err: Dict[str, int] = defaultdict(int)
    per_mod_tot: Dict[str, int] = defaultdict(int)
    for m in module_results:
        name = m.get("module", "unknown")
        per_mod_tot[name] += 1
        per_mod_lat[name].append(float(m.get("latency_ms", 0)))
        if any("error" in str(ev).lower() for ev in (m.get("evidence") or [])):
            per_mod_err[name] += 1
    for name, tot in per_mod_tot.items():
        snapshot[f"module_{name}_avg_latency_ms"] = (
            sum(per_mod_lat[name]) / tot if tot else 0.0
        )
        snapshot[f"module_{name}_error_rate"] = per_mod_err[name] / tot if tot else 0.0

    # Top-level error rate — includes both ErrorEvents and request volume.
    denom = max(1, total_requests)
    snapshot["error_rate"] = len(errors) / denom

    return snapshot


__all__ = [
    "Alert",
    "load_rules",
    "evaluate",
    "build_snapshot_from_events",
]
