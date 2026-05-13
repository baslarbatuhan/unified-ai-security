"""dashboard/lib/recommendations.py
=====================================
Hafta 13.2 — rule-based recommendations + composite security score.

The executive overview (1_home.py) calls into here twice per render:
once for `composite_security_score(metrics)` to put the headline number
in the header, and once for `generate_recommendations(metrics)` to
populate the bullet list below. Pure-fn: no Streamlit, no I/O. Lets us
test the policy decisions deterministically and ship the same logic
into reports later.

Score formula (v1 — hardcoded, documented; v2 can move to yaml):

    security_score = 100 * (1 − miss_rate)
                         * (1 − latency_breach_rate)
                         * detection_quality_factor

    detection_quality_factor = max(precision * recall, 0.0)

Bounded to [0, 100]. Missing inputs default to safe-neutral values
(no penalty) so a brand-new install starts at a useful baseline.

Recommendation rules trigger on the same metrics dict. Each returns a
`{severity, text}` dict suitable for `components.recommendation_panel`.
Severities pin to {info, warn, critical} so the panel's icon mapping
stays stable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Composite security score
# ---------------------------------------------------------------------------
_DEFAULT_SCORE_CONFIG: Dict[str, Any] = {
    "weights": {"miss": 1.0, "latency": 1.0, "quality": 1.0},
    "treat_zero_precision_recall_as_neutral": True,
    "max_score": 100.0,
}

_SCORE_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "security_score_weights.yaml"
)


def _clamp01(v: Any) -> float:
    """Coerce + clip to [0, 1]. None/garbage → 0.0."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, x))


def _load_score_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read `configs/security_score_weights.yaml`. Missing / unreadable
    config → defaults (so a fresh checkout / minimal install still
    produces sensible scores). Validated lightly: each weight must be
    a non-negative number; bad values fall back to the corresponding
    default."""
    p = path or _SCORE_CONFIG_PATH
    try:
        import yaml  # local import — keep recommendations.py importable
                     # even when yaml isn't installed (rare but cheap).
        if not p.exists():
            return dict(_DEFAULT_SCORE_CONFIG)
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception:  # noqa: BLE001 — fall back silently; never crash the score
        return dict(_DEFAULT_SCORE_CONFIG)

    cfg: Dict[str, Any] = dict(_DEFAULT_SCORE_CONFIG)
    # Merge weights (per-key fallback so partial yaml still works).
    yaml_weights = (raw.get("weights") or {})
    merged_weights = dict(_DEFAULT_SCORE_CONFIG["weights"])
    for k, default in merged_weights.items():
        v = yaml_weights.get(k)
        try:
            f = float(v)
            if f >= 0:
                merged_weights[k] = f
        except (TypeError, ValueError):
            pass  # keep default
    cfg["weights"] = merged_weights
    # Flags / scalars.
    if "treat_zero_precision_recall_as_neutral" in raw:
        cfg["treat_zero_precision_recall_as_neutral"] = bool(
            raw["treat_zero_precision_recall_as_neutral"]
        )
    try:
        ms = float(raw.get("max_score", cfg["max_score"]))
        if ms > 0:
            cfg["max_score"] = ms
    except (TypeError, ValueError):
        pass
    return cfg


def composite_security_score(
    metrics: Dict[str, Any],
    *,
    config: Optional[Dict[str, Any]] = None,
) -> float:
    """Compute the 0..max_score headline number.

    Required-ish keys (all optional, fall back to neutral):
        miss_rate            — float 0..1 (gateway_miss / total)
        latency_breach_rate  — float 0..1 (calls > module budget / total)
        precision, recall    — float 0..1 (block class)

    Formula (per `configs/security_score_weights.yaml`):
        score = max_score × (1 − miss)^w_miss
                          × (1 − latency_breach)^w_latency
                          × quality_factor^w_quality

    Weight = 0 disables the factor (it becomes 1 = neutral).
    Multiplicative on purpose: a regression in any one axis pulls the
    headline down so the executive sees it. Exponents let operators
    skew the formula toward their priorities without rewriting code.
    """
    cfg = config or _load_score_config()
    weights = cfg["weights"]
    treat_zero_neutral = bool(cfg["treat_zero_precision_recall_as_neutral"])
    max_score = float(cfg["max_score"])

    miss = _clamp01(metrics.get("miss_rate"))
    latency_breach = _clamp01(metrics.get("latency_breach_rate"))
    precision = _clamp01(metrics.get("precision"))
    recall = _clamp01(metrics.get("recall"))

    if precision == 0.0 and recall == 0.0 and treat_zero_neutral:
        quality_factor = 1.0
    else:
        quality_factor = precision * recall

    # Apply exponents. weight=0 → factor**0 = 1.0 (disabled cleanly).
    miss_factor = (1.0 - miss) ** float(weights["miss"])
    latency_factor = (1.0 - latency_breach) ** float(weights["latency"])
    quality_term = quality_factor ** float(weights["quality"])

    raw = max_score * miss_factor * latency_factor * quality_term
    return round(max(0.0, min(max_score, raw)), 1)


# ---------------------------------------------------------------------------
# Rule-based recommendations
# ---------------------------------------------------------------------------
# Threshold knobs are co-located so an analyst tuning the dashboard can
# change a single constant without hunting through if-statements.
HIGH_MISS_RATE = 0.10              # > 10% recall regression
MODERATE_MISS_RATE = 0.05
LATENCY_BUDGET_HOT_PCT = 0.80      # X module > 80% of timeout budget
LOW_ROUTING_SAVINGS_PCT = 30.0     # routing skips < 30% of chunks
HIGH_FLAG_BAND_RATIO = 0.50        # > 50% of blocks are flag-tier (suspicion)
LOW_AGENCY_ENGAGEMENT = 0.01       # agency_score == 0 across nearly all cases
HIGH_FALSE_POSITIVE_RATE = 0.05    # > 5% FP — over-blocking


def _miss_rate_recos(metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    miss = _clamp01(metrics.get("miss_rate"))
    if miss > HIGH_MISS_RATE:
        # `current` is whatever knob the analyst would lower — sanitize
        # threshold is the most common move. Suggestion is descriptive,
        # not prescriptive (we don't know the exact starting point).
        sanitize_thr = metrics.get("sanitize_threshold")
        suggest = (
            f" Consider lowering `sanitize` threshold from "
            f"{float(sanitize_thr):.2f} to ~{max(0.0, float(sanitize_thr) - 0.05):.2f}."
            if sanitize_thr is not None else
            " Consider lowering the `sanitize` / `block` thresholds in "
            "`configs/secure_balanced.yaml`."
        )
        out.append({
            "severity": "critical",
            "text": (
                f"Detection recall is below {(1 - HIGH_MISS_RATE) * 100:.0f}% "
                f"(miss rate {miss * 100:.1f}%).{suggest}"
            ),
        })
    elif miss > MODERATE_MISS_RATE:
        out.append({
            "severity": "warn",
            "text": (
                f"Detection recall is borderline "
                f"({(1 - miss) * 100:.1f}%). Watch the next runs — if it "
                "trends up, retune thresholds."
            ),
        })
    return out


def _latency_recos(metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    """Module-by-module latency vs configured budget. Expects
    `module_latency_ms` dict + `module_budget_ms` dict from caller."""
    out: List[Dict[str, str]] = []
    module_lat = metrics.get("module_latency_ms") or {}
    module_budget = metrics.get("module_budget_ms") or {}
    for mod, lat in module_lat.items():
        budget = float(module_budget.get(mod) or 0.0)
        if budget <= 0:
            continue
        ratio = float(lat or 0.0) / budget
        if ratio > LATENCY_BUDGET_HOT_PCT:
            out.append({
                "severity": "warn",
                "text": (
                    f"`{mod}` is using {ratio * 100:.0f}% of its timeout "
                    f"budget ({int(lat)} / {int(budget)} ms). "
                    "Consider raising `configs/timeout_config.yaml` budget "
                    "or optimising the pipeline."
                ),
            })
    return out


def _routing_recos(metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    skip_pct = metrics.get("routing_savings_pct")
    if skip_pct is None:
        return out
    try:
        skip_pct_f = float(skip_pct)
    except (TypeError, ValueError):
        return out
    if skip_pct_f < LOW_ROUTING_SAVINGS_PCT:
        out.append({
            "severity": "info",
            "text": (
                f"Chunk routing is skipping only {skip_pct_f:.1f}% of "
                "chunks — the embedding gate may be too permissive. "
                "Raise `RouterConfig.skip_below` slightly if false "
                "positives are not increasing."
            ),
        })
    return out


def _agency_engagement_recos(metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    """When the agency module's average score is ~0 across all cases,
    `tool_call` propagation is almost certainly broken upstream."""
    out: List[Dict[str, str]] = []
    avg_agency = metrics.get("avg_agency_score")
    n_agency_cases = metrics.get("n_agency_cases") or 0
    if avg_agency is None or int(n_agency_cases) == 0:
        return out
    if float(avg_agency) < LOW_AGENCY_ENGAGEMENT:
        out.append({
            "severity": "warn",
            "text": (
                "Agency module is not engaging — average agency_score is "
                f"{float(avg_agency):.3f} across {int(n_agency_cases)} cases. "
                "Verify `tool_call` propagation in the runner "
                "(see Hafta 10 fix pattern for rag_poisoning)."
            ),
        })
    return out


def _band_balance_recos(metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    """Most blocks are flag-tier (suspicion) → detection is not confident."""
    out: List[Dict[str, str]] = []
    flag_count = float(metrics.get("flag_band_count") or 0)
    block_count = float(metrics.get("block_band_count") or 0)
    total = flag_count + block_count
    if total < 5:
        return out  # too thin to recommend on
    flag_ratio = flag_count / total
    if flag_ratio > HIGH_FLAG_BAND_RATIO:
        out.append({
            "severity": "info",
            "text": (
                f"{flag_ratio * 100:.0f}% of recent blocks are flag-tier "
                "(suspicion). Detection signals could be sharper — review "
                "rag_guard and prompt_guard rule weights."
            ),
        })
    return out


def _false_positive_recos(metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    fp_rate = _clamp01(metrics.get("fp_rate"))
    if fp_rate > HIGH_FALSE_POSITIVE_RATE:
        out.append({
            "severity": "warn",
            "text": (
                f"False-positive rate is {fp_rate * 100:.1f}% — gateway is "
                "blocking legitimate input. Tighten patterns or raise the "
                "`block` threshold slightly."
            ),
        })
    return out


def generate_recommendations(
    metrics: Dict[str, Any],
    *,
    limit: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Run all rules; return ordered list of `{severity, text}` dicts.

    Order: critical → warn → info. Within a severity, rule order is the
    declaration order in `_RULES` (which roughly tracks "things that
    matter most when something looks off").
    """
    bag: List[Dict[str, str]] = []
    for rule in _RULES:
        try:
            bag.extend(rule(metrics))
        except Exception:  # noqa: BLE001 — never fail the dashboard for a rule bug
            continue
    # Sort by severity descending.
    sev_order = {"critical": 0, "warn": 1, "info": 2}
    bag.sort(key=lambda x: sev_order.get(str(x.get("severity") or "info"), 99))
    if limit is not None:
        bag = bag[:limit]
    return bag


_RULES = (
    _miss_rate_recos,
    _false_positive_recos,
    _latency_recos,
    _agency_engagement_recos,
    _band_balance_recos,
    _routing_recos,
)


__all__ = [
    "composite_security_score",
    "generate_recommendations",
    "HIGH_MISS_RATE",
    "MODERATE_MISS_RATE",
    "LATENCY_BUDGET_HOT_PCT",
    "LOW_ROUTING_SAVINGS_PCT",
    "HIGH_FLAG_BAND_RATIO",
    "LOW_AGENCY_ENGAGEMENT",
    "HIGH_FALSE_POSITIVE_RATE",
]
