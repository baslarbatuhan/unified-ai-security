"""dashboard/lib/components.py
================================
Hafta 13 — reusable Streamlit widgets shared across pages.

Six small primitives that the executive overview (1_home.py) and the
analyst pages (Live monitor / Results / Compare runs) all build on:

    run_selector_widget()      — registry-driven run picker + URL sync
    decision_card()            — coloured per-module risk card
    risk_gauge()               — 0..1 bar with threshold markers
    executive_summary_panel()  — top-line security score + risk level
    evidence_table()           — formatted evidence list
    recommendation_panel()     — bullet list with severity icons

Each function takes pre-computed inputs (numbers / strings / lists) and
calls Streamlit primitives. We deliberately keep the data-shaping logic
out of the components — callers compose. That way the components stay
pure-fn from a testing perspective (signature checks, label formatting)
and the data layer can evolve without breaking the UI surface.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Colour mapping for decision-bearing widgets. Picked to stay legible on
# Streamlit's default light/dark themes; matches what tests assert against.
# ---------------------------------------------------------------------------
DECISION_COLORS: Dict[str, str] = {
    "allow":    "#22c55e",  # green-500
    "sanitize": "#eab308",  # yellow-500
    "flag":     "#f97316",  # orange-500
    "block":    "#ef4444",  # red-500
}
DECISION_ICONS: Dict[str, str] = {
    "allow":    "✅",
    "sanitize": "⚠️",
    "flag":     "🟠",
    "block":    "⛔",
}


def decision_color(decision: str) -> str:
    """Public lookup so other pages can colour rows in their own widgets
    without reaching into the dict directly (decouples the palette)."""
    return DECISION_COLORS.get(str(decision).lower(), "#94a3b8")  # slate-400 fallback


def decision_icon(decision: str) -> str:
    return DECISION_ICONS.get(str(decision).lower(), "ℹ️")


# ---------------------------------------------------------------------------
# 1) run_selector_widget — used on Live monitor, Results, Compare runs
# ---------------------------------------------------------------------------
def format_registry_label(entry: Dict[str, Any]) -> str:
    """Shape a registry entry into a single line for selectbox options.

    Same format used on every page so users see consistent identifiers:
        "<ended_at> · <target> · <suite> · n=<rows> · <run_id>"
    """
    rid = entry.get("run_id") or "?"
    tgt = entry.get("target_id") or "?"
    suite = entry.get("suite") or "?"
    n = entry.get("n_rows") or entry.get("n_cases") or 0
    when = (entry.get("ended_at") or entry.get("started_at") or "")[:19].replace("T", " ")
    return f"{when} · {tgt} · {suite} · n={n} · {rid}"


def run_selector_widget(
    registry: List[Dict[str, Any]],
    *,
    key: str,
    default_run_id: Optional[str] = None,
    label: str = "Run",
    qp_field: str = "run_id",
) -> Optional[Dict[str, Any]]:
    """Render a selectbox over a registry list and return the picked entry.

    Args:
        registry: List of registry entries (newest-first). Empty → returns None.
        key: Unique Streamlit widget key so multiple selectors can coexist.
        default_run_id: If provided AND present in registry, picks this one
            on first render. Otherwise honours `?<qp_field>=...` query
            param (deep link). Otherwise defaults to the newest entry.
        label: Visible label above the selectbox.
        qp_field: URL query-param name used for deep-linking the selection.

    Returns:
        The full registry dict for the chosen run, or None when empty.
    """
    if not registry:
        st.warning("No runs registered yet. Launch a run from the Run test page.")
        return None

    labels = [format_registry_label(e) for e in registry]
    # Resolve default index: explicit arg → query param → first row.
    qp_run_id = st.query_params.get(qp_field) if qp_field else None
    default_idx = 0
    if default_run_id:
        for i, e in enumerate(registry):
            if e.get("run_id") == default_run_id:
                default_idx = i
                break
    elif qp_run_id:
        for i, e in enumerate(registry):
            if e.get("run_id") == qp_run_id:
                default_idx = i
                break

    picked_label = st.selectbox(label, labels, index=default_idx, key=key)
    picked_idx = labels.index(picked_label)
    entry = registry[picked_idx]
    if qp_field:
        rid = entry.get("run_id") or ""
        if rid:
            st.query_params[qp_field] = rid
    return entry


# ---------------------------------------------------------------------------
# 2) decision_card — coloured per-module card for trace drill-down
# ---------------------------------------------------------------------------
def decision_card(
    module: str,
    risk: float,
    decision: str,
    evidence: Optional[Sequence[str]] = None,
    threshold: Optional[float] = None,
    *,
    container: Any = None,
) -> None:
    """Render a coloured card for one module's verdict.

    The card is intentionally minimal — a header with the decision icon +
    risk score, optional threshold, and the top-3 evidence strings. We
    use `st.markdown(...)` with an inline-styled <div> rather than
    `st.metric` so the colour band wraps the whole card (legible at a
    glance during demo).
    """
    tgt = container or st
    color = decision_color(decision)
    icon = decision_icon(decision)
    risk_pct = f"{float(risk) * 100:.1f}%" if risk is not None else "—"
    threshold_str = f" / thr {float(threshold):.2f}" if threshold is not None else ""
    body_lines = []
    if evidence:
        for line in list(evidence)[:3]:
            # Escape any markdown control chars in evidence — analysts
            # paste raw strings here and we don't want `*` or `_` to
            # reformat the card.
            safe = str(line).replace("|", "│")[:160]
            body_lines.append(f"<div style='font-size:0.85em;margin-top:2px'>• {safe}</div>")
    body_html = "".join(body_lines) or (
        "<div style='font-size:0.85em;margin-top:2px;opacity:0.6'>(no evidence)</div>"
    )
    card_html = (
        f"<div style='padding:10px 14px;border-left:4px solid {color};"
        f"background:rgba(127,127,127,0.05);border-radius:4px;margin-bottom:8px'>"
        f"<div style='font-weight:600;font-size:1.0em'>"
        f"{icon} {module} — <span style='color:{color}'>{decision}</span> "
        f"<span style='opacity:0.75'>({risk_pct}{threshold_str})</span>"
        f"</div>"
        f"{body_html}"
        f"</div>"
    )
    tgt.markdown(card_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 3) risk_gauge — 0..1 bar with threshold markers
# ---------------------------------------------------------------------------
def risk_gauge(
    score: float,
    thresholds: Optional[Dict[str, float]] = None,
    *,
    label: str = "fused_risk",
    container: Any = None,
) -> None:
    """Inline gauge — score as a coloured bar, threshold lines overlaid.

    Streamlit doesn't ship a native gauge widget; we render an HTML bar
    so the threshold markers (allow / sanitize / block) are positioned
    by % CSS, which the dashboard's existing renderer handles cleanly.
    """
    tgt = container or st
    s = max(0.0, min(1.0, float(score or 0.0)))
    thr = thresholds or {"allow": 0.3, "sanitize": 0.6, "block": 0.85}
    pct = s * 100.0
    # Pick the colour from the band the score falls into.
    if s >= thr.get("block", 0.85):
        bar_color = DECISION_COLORS["block"]
    elif s >= thr.get("sanitize", 0.6):
        bar_color = DECISION_COLORS["flag"]
    elif s >= thr.get("allow", 0.3):
        bar_color = DECISION_COLORS["sanitize"]
    else:
        bar_color = DECISION_COLORS["allow"]

    marker_html = "".join(
        f"<div style='position:absolute;top:-4px;left:{float(v)*100:.1f}%;"
        f"width:1px;height:22px;background:#64748b;'></div>"
        for v in thr.values()
    )
    html = (
        f"<div style='font-size:0.85em;margin-bottom:4px'>"
        f"{label}: <b>{s:.3f}</b></div>"
        f"<div style='position:relative;height:14px;background:rgba(127,127,127,0.10);"
        f"border-radius:7px;overflow:visible'>"
        f"<div style='height:14px;width:{pct:.1f}%;background:{bar_color};"
        f"border-radius:7px'></div>"
        f"{marker_html}"
        f"</div>"
    )
    tgt.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 4) executive_summary_panel — top-line security score + risk level
# ---------------------------------------------------------------------------
def risk_level_from_score(score: float) -> str:
    """Map composite security score (0-100) to a 4-tier risk label.

    Inverted: high score = low risk. Bands are tuned so a freshly-tuned
    system (recall ~95% + acceptable FP) lands in LOW; recall < 80% or
    repeated latency breaches push into HIGH/CRITICAL.
    """
    s = float(score or 0.0)
    if s >= 85:
        return "LOW"
    if s >= 70:
        return "MEDIUM"
    if s >= 50:
        return "HIGH"
    return "CRITICAL"


_RISK_COLORS = {
    "LOW":      "#22c55e",
    "MEDIUM":   "#eab308",
    "HIGH":     "#f97316",
    "CRITICAL": "#ef4444",
}


def executive_summary_panel(
    metrics: Dict[str, Any],
    *,
    container: Any = None,
) -> None:
    """Top-of-page header for the executive overview.

    Expected `metrics` keys (all optional; missing → "—"):
        security_score      — composite 0..100 (caller computes)
        n_recent_runs       — how many runs the score covers
        miss_rate           — float 0..1
        block_rate          — float 0..1
        avg_latency_ms      — float
        precision, recall, f1 — float 0..1
    """
    tgt = container or st
    score = float(metrics.get("security_score") or 0.0)
    level = risk_level_from_score(score)
    level_color = _RISK_COLORS.get(level, "#94a3b8")

    n_runs = metrics.get("n_recent_runs", 0)
    miss = float(metrics.get("miss_rate") or 0.0)
    block = float(metrics.get("block_rate") or 0.0)
    avg_lat = float(metrics.get("avg_latency_ms") or 0.0)
    precision = float(metrics.get("precision") or 0.0)
    recall = float(metrics.get("recall") or 0.0)
    f1 = float(metrics.get("f1") or 0.0)

    # Header bar — score badge + risk level badge.
    tgt.markdown(
        f"<div style='display:flex;align-items:center;gap:24px;padding:14px 16px;"
        f"border-radius:8px;background:rgba(127,127,127,0.07);margin-bottom:12px'>"
        f"<div><div style='font-size:0.8em;opacity:0.7'>Security score</div>"
        f"<div style='font-size:2.2em;font-weight:700'>{score:.0f}<span style='font-size:0.5em;opacity:0.6'> / 100</span></div></div>"
        f"<div><div style='font-size:0.8em;opacity:0.7'>Risk level</div>"
        f"<div style='font-size:1.6em;font-weight:700;color:{level_color}'>{level}</div></div>"
        f"<div style='margin-left:auto;font-size:0.85em;opacity:0.6'>covers last {int(n_runs)} runs</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Metric row — concrete numbers behind the score.
    cols = tgt.columns(6)
    cols[0].metric("Miss rate", f"{miss * 100:.1f} %",
                   help="Gateway returned allow on a case expected to block/sanitize.")
    cols[1].metric("Block rate", f"{block * 100:.1f} %")
    cols[2].metric("Avg latency", f"{avg_lat:.0f} ms")
    cols[3].metric("Precision", f"{precision:.3f}")
    cols[4].metric("Recall", f"{recall:.3f}")
    cols[5].metric("F1", f"{f1:.3f}")


# ---------------------------------------------------------------------------
# 5) evidence_table — formatted evidence display
# ---------------------------------------------------------------------------
def evidence_table(
    rows: Iterable[Dict[str, Any]],
    *,
    columns: Optional[List[str]] = None,
    container: Any = None,
) -> None:
    """Render a list-of-dict as a pandas table. Empty → friendly hint.

    `columns` restricts/orders displayed fields when provided; otherwise
    the dict's natural keys ship as-is.
    """
    tgt = container or st
    rows_list = list(rows or [])
    if not rows_list:
        tgt.caption("(no evidence rows)")
        return
    df = pd.DataFrame(rows_list)
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    tgt.dataframe(df, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# 6) recommendation_panel — bullet list with severity icons
# ---------------------------------------------------------------------------
SEVERITY_ICONS: Dict[str, str] = {
    "info":     "ℹ️",
    "warn":     "⚠️",
    "critical": "🚨",
}


def recommendation_panel(
    recos: List[Dict[str, str]],
    *,
    title: str = "Recommendations",
    container: Any = None,
) -> None:
    """Render structured recommendations as a bulleted list.

    Each item is `{severity: 'info'|'warn'|'critical', text: '...'}`. A
    plain list[str] is accepted too for callers that haven't graduated
    to structured items yet.
    """
    tgt = container or st
    if not recos:
        tgt.caption(f"_{title}: nothing to flag right now._")
        return
    tgt.subheader(title)
    for item in recos:
        if isinstance(item, dict):
            sev = str(item.get("severity") or "info").lower()
            text = str(item.get("text") or "").strip()
        else:
            sev = "info"
            text = str(item)
        if not text:
            continue
        icon = SEVERITY_ICONS.get(sev, "•")
        tgt.markdown(f"{icon}  {text}")


__all__ = [
    "DECISION_COLORS",
    "DECISION_ICONS",
    "SEVERITY_ICONS",
    "decision_color",
    "decision_icon",
    "format_registry_label",
    "run_selector_widget",
    "decision_card",
    "risk_gauge",
    "risk_level_from_score",
    "executive_summary_panel",
    "evidence_table",
    "recommendation_panel",
]
