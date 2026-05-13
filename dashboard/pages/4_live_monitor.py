"""Live monitor — per-module risk and latency tail.

Polls the gateway's read-only routes and overlays the launched run
status (if any) so the demo stays in sync with what's actually happening
in the background subprocess.

Two scopes:
  * Global  — every request the gateway has seen in the recent window
              (default; same behaviour the page always had).
  * This run — narrows summary/recent feeds to a specific run_id so
              the metrics on screen match the banner. Solves the
              "banner says one run, charts show all traffic" mismatch.

The selected run_id is also pushed into the URL (`?run_id=...`) and
mirrored into `st.session_state["last_run_id"]`, so opening the page
fresh — new tab, page reload, or shared link — keeps the run focused.
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


try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except ImportError:  # pragma: no cover — optional dep
    st_autorefresh = None


st.set_page_config(page_title="Live monitor", page_icon=":satellite:", layout="wide")
st.title("Live monitor")
client = get_default_client()


# ---------------------------------------------------------------------------
# Resolve run_id: URL query param wins (shareable link), else session_state
# (Run test page set it on launch). Session_state stays the canonical store.
# ---------------------------------------------------------------------------
qp_run_id = st.query_params.get("run_id")
if qp_run_id and qp_run_id != st.session_state.get("last_run_id"):
    st.session_state["last_run_id"] = qp_run_id

last_run_id = st.session_state.get("last_run_id")


# ---------------------------------------------------------------------------
# Top controls — refresh + window + scope
# ---------------------------------------------------------------------------
col_a, col_b, col_c = st.columns([1, 2, 3])
refresh_s = col_a.number_input("Refresh (s)", min_value=2, max_value=30, value=5, step=1)
window = col_b.slider("Window (recent decisions)", 10, 200, 50, 10)

# Scope: Global vs This run. Default "This run" if a run is tracked, so the
# banner and charts agree out of the box.
scope_options = ["Global"]
if last_run_id:
    scope_options.append(f"This run ({last_run_id})")
default_idx = 1 if last_run_id else 0
scope = col_c.radio(
    "Scope",
    scope_options,
    index=default_idx,
    horizontal=True,
    help="Global aggregates every request in the window. "
         "'This run' filters summary + recent decisions to the launched run_id.",
)
scoped_run_id = last_run_id if scope.startswith("This run") else None

# Keep the URL in sync so reloading or sharing the link preserves the scope.
if scoped_run_id:
    st.query_params["run_id"] = scoped_run_id
elif "run_id" in st.query_params:
    del st.query_params["run_id"]

if st_autorefresh is not None:
    st_autorefresh(interval=int(refresh_s * 1000), key="live_monitor_refresh")
else:
    st.info("Install `streamlit-autorefresh` for hands-free polling. Hit `R` to refresh manually.")

# ---------------------------------------------------------------------------
# Active run banner
# ---------------------------------------------------------------------------
if last_run_id:
    try:
        status = client.get_json(f"/runs/{last_run_id}/status") or {}
    except GatewayError:
        status = {}
    state = status.get("state", "?")
    badge = {"queued": ":hourglass:", "running": ":runner:", "done": ":white_check_mark:", "failed": ":x:"}.get(state, "")
    st.info(f"{badge} Tracked run `{last_run_id}` — state: **{state}** ({status.get('updated_at', '')})")

# ---------------------------------------------------------------------------
# Top-line counters — scope-aware
# ---------------------------------------------------------------------------
summary_params = {"run_id": scoped_run_id} if scoped_run_id else None
recent_params = {"limit": window}
if scoped_run_id:
    recent_params["run_id"] = scoped_run_id

try:
    summary = client.get_json("/dashboard/summary", params=summary_params) or {}
    recent = client.get_json("/dashboard/recent-runs", params=recent_params) or {}
except GatewayError as exc:
    st.error(f"Gateway unreachable: {exc}")
    st.stop()

total = int(summary.get("total_requests", 0) or 0)
block_n = int(summary.get("block_count", 0) or 0)
sanitize_n = int(summary.get("sanitize_count", 0) or 0)
allow_n = int(summary.get("allow_count", 0) or 0)
# Fallback: API may return only rates; compute counts locally.
if block_n == 0 and total > 0:
    block_n = int(round(float(summary.get("block_rate", 0.0) or 0.0) * total))
if sanitize_n == 0 and total > 0:
    sanitize_n = int(round(float(summary.get("sanitize_rate", 0.0) or 0.0) * total))
if allow_n == 0 and total > 0:
    allow_n = int(round(float(summary.get("allow_rate", 0.0) or 0.0) * total))
# avg_total_latency_ms is the alias; fall back to avg_latency_ms if not present.
avg_lat_ms = float(
    summary.get("avg_total_latency_ms") or summary.get("avg_latency_ms") or 0
)

scope_label = f"This run · `{scoped_run_id}`" if scoped_run_id else "Global"
st.caption(f"Scope: {scope_label}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total requests", total)
c2.metric("Block count", block_n)
c3.metric("Sanitize count", sanitize_n)
c4.metric("Allow count", allow_n)
c5.metric("Avg latency", f"{avg_lat_ms:.0f} ms")

# ---------------------------------------------------------------------------
# Per-module average latency (which module is heaviest right now)
# ---------------------------------------------------------------------------
mod_lat = []
for mod in ("prompt_guard", "rag_guard", "output_agency", "output_guard"):
    val = summary.get(f"module_{mod}_avg_latency_ms")
    if val is not None:
        mod_lat.append({"module": mod, "avg_latency_ms": float(val)})
if mod_lat:
    st.subheader("Per-module latency")
    st.bar_chart(pd.DataFrame(mod_lat).set_index("module"))
    # Caveat: external_eval subprocess writes FusionDecisionEvent but not
    # ModuleResultEvent, so this chart can be sparse during suite runs.
    st.caption(
        "ℹ️ Built from `module_result` telemetry. External-eval suite runs "
        "primarily emit `fusion_decision` events; per-module latencies may "
        "be sparse or missing for those windows."
    )

# ---------------------------------------------------------------------------
# Per-module risk in the recent window — which module produced the score?
# ---------------------------------------------------------------------------
items = recent.get("decisions") or []
if items:
    df = pd.DataFrame(items)
    score_cols = [c for c in ("prompt_score", "rag_score", "agency_score", "output_score") if c in df.columns]
    if score_cols:
        st.subheader(f"Per-module risk in last {len(df)} decisions")
        st.line_chart(df[score_cols])

    st.subheader("Recent decisions")
    st.dataframe(df, width="stretch", hide_index=True)

    # Hafta 12.1: trace drill-down — only meaningful when we're scoped
    # to a specific run (the trace file is per-run).
    if scoped_run_id and "attack_id" in df.columns:
        st.markdown("**🔍 Decision trace**")
        cases = df["attack_id"].dropna().astype(str).unique().tolist()
        if cases:
            picked = st.selectbox(
                "Pick a case_id to inspect", cases,
                key=f"live_trace_case::{scoped_run_id}",
            )
            if picked:
                try:
                    tr = client.get_json(
                        f"/decisions/{scoped_run_id}/{picked}/trace"
                    ) or {}
                    trace = tr.get("trace") or {}
                except GatewayError:
                    trace = {}
                if trace:
                    cc = st.columns(4)
                    cc[0].metric("final_decision", trace.get("final_decision", "?"))
                    cc[1].metric("decision_band", trace.get("decision_band", "?"))
                    cc[2].metric("fused_risk", trace.get("fused_risk", "?"))
                    cc[3].metric("triggering", trace.get("triggering_module", "?"))
                    st.code(trace.get("fusion_formula", ""), language="text")
                else:
                    st.caption(
                        "No trace yet — runs from before Hafta 12.1 won't have one."
                    )
else:
    if scoped_run_id:
        st.info(
            f"No fusion decisions for run `{scoped_run_id}` in this window — "
            "the run may still be warming up, or its decisions fell outside "
            "the recent telemetry window. Try widening the window or switching "
            "to the Global scope."
        )
    else:
        st.info("No fusion decisions in this window.")
