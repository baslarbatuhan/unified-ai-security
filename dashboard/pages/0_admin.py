"""Admin — operational console (Hafta 13).

Streamlit orders pages by filename prefix, so `0_` puts this above
`1_home`. Intent: every gauge that's useful to a SRE / on-call engineer
but distracting for a non-technical reviewer lives here. The executive
overview on `1_home.py` consumes nothing from this page directly —
they share the same gateway endpoints but ship to different audiences.

Sections:
    /health full snapshot         — 5 system checks, raw JSON inspectable
    Circuit breakers              — state + counters per registered breaker
    Rate limiter levels           — current token-bucket fill
    /dashboard/summary raw        — last 500 telemetry events aggregated
    Recent telemetry tail         — last 50 jsonl events (compact)

Nothing here is interactive beyond Streamlit's auto-refresh; this page
is read-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from dashboard.lib.gateway_client import GatewayError, get_default_client


st.set_page_config(page_title="Admin", page_icon=":gear:", layout="wide")
st.title("Admin — operational console")
st.caption(
    "Technical detail surface for SRE / on-call use. The Executive "
    "overview lives on **Home**. Everything below is read-only — no "
    "writes hit the gateway from this page."
)

client = get_default_client()


# ---------------------------------------------------------------------------
# 1) /health snapshot — the same JSON the gateway's startup check emits
# ---------------------------------------------------------------------------
st.subheader("Health")
try:
    health = client.get_json("/health") or {}
except GatewayError as exc:
    st.error(f"Gateway unreachable: {exc}")
    st.stop()

status = str(health.get("status") or "?")
passed = int(health.get("passed") or 0)
total = int(health.get("total") or 0)
hc1, hc2 = st.columns([1, 3])
hc1.metric("Status", status, delta=f"{passed}/{total} checks")
checks = health.get("checks") or []
if checks:
    hc2.dataframe(pd.DataFrame(checks), width="stretch", hide_index=True)
with st.expander("Raw /health JSON"):
    st.json(health)


# ---------------------------------------------------------------------------
# 2) Circuit breakers
# ---------------------------------------------------------------------------
st.subheader("Circuit breakers")
try:
    cb = client.get_json("/dashboard/circuit-breakers") or {}
except GatewayError as exc:
    cb = {}
    st.warning(f"Could not load circuit breakers: {exc}")
breakers = cb.get("breakers") or []
if breakers:
    st.dataframe(pd.DataFrame(breakers), width="stretch", hide_index=True)
    st.caption(
        "States: `closed` = normal; `open` = short-circuiting calls; "
        "`half_open` = probing. Hafta 11 wired the `ollama_llm_judge` "
        "breaker into the RAG judge path."
    )
else:
    st.caption("No breakers registered yet — no consumer has called `get_breaker(...)`.")


# ---------------------------------------------------------------------------
# 3) Rate limiter
# ---------------------------------------------------------------------------
st.subheader("Rate limiter")
try:
    rl = client.get_json("/dashboard/rate-limits") or {}
except GatewayError:
    rl = {}
buckets = rl.get("buckets") or []
if buckets:
    st.dataframe(pd.DataFrame(buckets), width="stretch", hide_index=True)
else:
    st.caption("No active rate-limit buckets.")


# ---------------------------------------------------------------------------
# 4) /dashboard/summary aggregated snapshot
# ---------------------------------------------------------------------------
st.subheader("Aggregated summary (last 500 events)")
try:
    summary = client.get_json("/dashboard/summary") or {}
except GatewayError as exc:
    summary = {}
    st.warning(f"Could not load summary: {exc}")
if summary:
    # Pull a useful subset to the top so the eye lands on it.
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Total requests", int(summary.get("total_requests") or 0))
    sc2.metric("Block rate", f"{float(summary.get('block_rate') or 0) * 100:.1f}%")
    sc3.metric("Sanitize rate", f"{float(summary.get('sanitize_rate') or 0) * 100:.1f}%")
    sc4.metric("Allow rate", f"{float(summary.get('allow_rate') or 0) * 100:.1f}%")

    mod_rows = []
    for mod in ("prompt_guard", "rag_guard", "output_agency", "output_guard"):
        avg_lat = summary.get(f"module_{mod}_avg_latency_ms")
        blocks = summary.get(f"module_{mod}_block_count")
        if avg_lat is None and blocks is None:
            continue
        mod_rows.append({
            "module": mod, "avg_latency_ms": avg_lat, "block_count": blocks,
        })
    if mod_rows:
        st.markdown("**Per-module averages**")
        st.dataframe(pd.DataFrame(mod_rows), width="stretch", hide_index=True)

    with st.expander("Raw summary JSON"):
        st.json(summary)


# ---------------------------------------------------------------------------
# 5) Recent telemetry tail — drilldown for ops
# ---------------------------------------------------------------------------
st.subheader("Recent telemetry")
try:
    recent = client.get_json("/dashboard/recent-runs", params={"limit": 30}) or {}
except GatewayError as exc:
    recent = {}
    st.warning(f"Could not load recent decisions: {exc}")
items = recent.get("decisions") or []
if items:
    df = pd.DataFrame(items)
    keep = [c for c in (
        "event_id", "run_id", "timestamp", "kind", "target_id", "attack_id",
        "fused_risk_score", "decision", "prompt_score", "rag_score",
        "agency_score", "output_score", "latency_ms_total",
    ) if c in df.columns]
    st.dataframe(df[keep] if keep else df, width="stretch", hide_index=True)
else:
    st.caption("No fusion decisions in the last window.")


st.divider()
st.caption(
    "ℹ️ This page is intended for technical operators. For the "
    "executive-level view, open **Home** at the top of the navigation."
)
