"""reporting/summary_generator.py
Executive summary — boils a metrics snapshot down to the fields a non-technical
reader cares about: how many requests, what fraction were blocked, the bypass
proxy, p95 latency, top alert, and the strongest/weakest module by latency.

The two callers we care about:
    * report_generator (end-to-end markdown assembly)
    * dashboard back-end (cheap "headline" string for the home page)

Pure functions, no I/O. Tests feed snapshot dicts directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Summary:
    """Headline fields plus the raw snapshot for callers that want more."""

    total_requests: int
    block_rate: float
    sanitize_rate: float
    allow_rate: float
    flag_rate: float
    bypass_rate: float
    p95_latency_ms: float
    avg_latency_ms: float
    error_rate: float
    top_alert: Optional[Dict[str, Any]] = None
    slowest_module: Optional[str] = None
    slowest_module_avg_ms: float = 0.0
    fastest_module: Optional[str] = None
    fastest_module_avg_ms: float = 0.0
    # "strongest" = module that contributed risk most often (block_count fallback
    # to high-score count). Detection-quality answer to "en güçlü modül".
    strongest_module: Optional[str] = None
    strongest_module_block_count: int = 0
    snapshot: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["snapshot"] = dict(self.snapshot)
        return d


def _module_latency_extremes(snapshot: Dict[str, float]):
    """Return (slowest_name, slowest_ms, fastest_name, fastest_ms) over modules."""
    pairs: List = []
    for k, v in snapshot.items():
        if k.startswith("module_") and k.endswith("_avg_latency_ms"):
            name = k[len("module_"): -len("_avg_latency_ms")]
            try:
                pairs.append((name, float(v)))
            except (TypeError, ValueError):
                continue
    # Filter modules that never ran (latency 0 means no data, not "fast").
    active = [p for p in pairs if p[1] > 0]
    if not active:
        return None, 0.0, None, 0.0
    active.sort(key=lambda p: p[1])
    fastest_name, fastest_ms = active[0]
    slowest_name, slowest_ms = active[-1]
    return slowest_name, slowest_ms, fastest_name, fastest_ms


def _strongest_module(snapshot: Dict[str, float]):
    """Pick the module that contributed risk most often.

    Order of preference: explicit `module_<name>_block_count`, then
    `module_<name>_high_score_count`, then None. Detection-quality metric —
    distinct from latency-based "fastest/slowest".
    """
    counts: List = []
    for k, v in snapshot.items():
        if k.startswith("module_") and (
            k.endswith("_block_count") or k.endswith("_high_score_count")
        ):
            suffix_len = len("_block_count") if k.endswith("_block_count") else len("_high_score_count")
            name = k[len("module_"): -suffix_len]
            try:
                counts.append((name, int(float(v))))
            except (TypeError, ValueError):
                continue
    counts = [(n, c) for n, c in counts if c > 0]
    if not counts:
        return None, 0
    counts.sort(key=lambda p: -p[1])
    return counts[0]


def _flag_rate(snapshot: Dict[str, float]) -> float:
    """Snapshot tracks block/sanitize/allow but not flag explicitly. Derive it."""
    total = float(snapshot.get("total_requests", 0.0) or 0.0)
    if total <= 0:
        return 0.0
    accounted = (
        float(snapshot.get("block_rate", 0.0))
        + float(snapshot.get("sanitize_rate", 0.0))
        + float(snapshot.get("allow_rate", 0.0))
    )
    # "flag" is whatever isn't block/sanitize/allow. Clamp to >= 0 in case
    # rounding overshoots.
    return max(0.0, 1.0 - accounted)


def build_summary(
    snapshot: Dict[str, float],
    alerts: Optional[List[Dict[str, Any]]] = None,
) -> Summary:
    """Pure transform: snapshot dict → Summary dataclass."""
    alerts = alerts or []
    severity_order = {"critical": 2, "warn": 1, "info": 0}
    top_alert = None
    if alerts:
        top_alert = max(alerts, key=lambda a: severity_order.get(a.get("severity", "info"), 0))

    slow_name, slow_ms, fast_name, fast_ms = _module_latency_extremes(snapshot)
    strong_name, strong_count = _strongest_module(snapshot)

    return Summary(
        total_requests=int(snapshot.get("total_requests", 0) or 0),
        block_rate=float(snapshot.get("block_rate", 0.0) or 0.0),
        sanitize_rate=float(snapshot.get("sanitize_rate", 0.0) or 0.0),
        allow_rate=float(snapshot.get("allow_rate", 0.0) or 0.0),
        flag_rate=_flag_rate(snapshot),
        bypass_rate=float(snapshot.get("attack_bypass_rate", 0.0) or 0.0),
        p95_latency_ms=float(snapshot.get("p95_latency_ms", 0.0) or 0.0),
        avg_latency_ms=float(snapshot.get("avg_latency_ms", 0.0) or 0.0),
        error_rate=float(snapshot.get("error_rate", 0.0) or 0.0),
        top_alert=top_alert,
        slowest_module=slow_name,
        slowest_module_avg_ms=slow_ms,
        fastest_module=fast_name,
        fastest_module_avg_ms=fast_ms,
        strongest_module=strong_name,
        strongest_module_block_count=strong_count,
        snapshot=dict(snapshot),
    )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def _pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def render_summary(summary: Summary) -> str:
    """Compose the executive summary markdown block."""
    lines: List[str] = []
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- **Requests evaluated:** {summary.total_requests}")
    lines.append(f"- **Block rate:** {_pct(summary.block_rate)}  "
                 f"· **Sanitize:** {_pct(summary.sanitize_rate)}  "
                 f"· **Flag:** {_pct(summary.flag_rate)}  "
                 f"· **Allow:** {_pct(summary.allow_rate)}")
    lines.append(f"- **Bypass proxy** (module ≥ 0.60 yet allowed): "
                 f"**{_pct(summary.bypass_rate)}**")
    lines.append(f"- **Latency:** avg {summary.avg_latency_ms:.0f} ms · "
                 f"p95 {summary.p95_latency_ms:.0f} ms")
    if summary.error_rate > 0:
        lines.append(f"- **Error rate:** {_pct(summary.error_rate, 2)}")

    if summary.slowest_module and summary.fastest_module:
        lines.append(
            f"- **Module latency:** slowest `{summary.slowest_module}` "
            f"({summary.slowest_module_avg_ms:.0f} ms) · "
            f"fastest `{summary.fastest_module}` "
            f"({summary.fastest_module_avg_ms:.0f} ms)"
        )

    if summary.strongest_module:
        lines.append(
            f"- **Strongest module:** `{summary.strongest_module}` "
            f"(contributed to {summary.strongest_module_block_count} blocks)"
        )

    if summary.top_alert:
        a = summary.top_alert
        lines.append(
            f"- **Top alert:** `{a.get('rule_id', '?')}` "
            f"({a.get('severity', '?')}) — {a.get('message', '')}"
        )
    else:
        lines.append("- **Top alert:** none active in window")

    lines.append("")
    return "\n".join(lines)


__all__ = ["Summary", "build_summary", "render_summary"]
