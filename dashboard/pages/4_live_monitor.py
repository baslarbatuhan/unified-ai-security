"""Live monitor — per-module risk and latency tail.

Polls the gateway's read-only routes and overlays the launched run
status (if any) so the demo stays in sync with what's actually happening
in the background subprocess.
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


col_a, col_b = st.columns([1, 5])
refresh_s = col_a.number_input("Refresh (s)", min_value=2, max_value=30, value=5, step=1)
window = col_b.slider("Window (recent decisions)", 10, 200, 50, 10)

if st_autorefresh is not None:
    st_autorefresh(interval=int(refresh_s * 1000), key="live_monitor_refresh")
else:
    st.info("Install `streamlit-autorefresh` for hands-free polling. Hit `R` to refresh manually.")

# ---------------------------------------------------------------------------
# Active run banner
# ---------------------------------------------------------------------------
last_run_id = st.session_state.get("last_run_id")
if last_run_id:
    try:
        status = client.get_json(f"/runs/{last_run_id}/status") or {}
    except GatewayError:
        status = {}
    state = status.get("state", "?")
    badge = {"queued": ":hourglass:", "running": ":runner:", "done": ":white_check_mark:", "failed": ":x:"}.get(state, "")
    st.info(f"{badge} Tracked run `{last_run_id}` — state: **{state}** ({status.get('updated_at', '')})")

# ---------------------------------------------------------------------------
# Top-line counters
# ---------------------------------------------------------------------------
try:
    summary = client.get_json("/dashboard/summary") or {}
    recent = client.get_json("/dashboard/recent-runs", params={"limit": window}) or {}
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
else:
    st.info("No fusion decisions in this window.")
