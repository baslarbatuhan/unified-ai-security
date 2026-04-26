"""Home — single-glance system snapshot."""
from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from dashboard.lib.gateway_client import GatewayError, get_default_client


st.set_page_config(page_title="Home", page_icon=":house:", layout="wide")
st.title("Home")
client = get_default_client()


# ---------------------------------------------------------------------------
# Top-line summary
# ---------------------------------------------------------------------------
try:
    summary = client.get_json("/dashboard/summary") or {}
    health = client.get_json("/health") or {}
except GatewayError as exc:
    st.error(f"Gateway unreachable: {exc}")
    st.stop()

# "Detection rate" = anything not-allow (block + sanitize + flag).
# Single-glance success metric the spec asks for ("genel başarı oranı").
block_r = float(summary.get("block_rate", 0.0) or 0.0)
sanitize_r = float(summary.get("sanitize_rate", 0.0) or 0.0)
allow_r = float(summary.get("allow_rate", 0.0) or 0.0)
flag_r = max(0.0, 1.0 - block_r - sanitize_r - allow_r)
detection_rate = block_r + sanitize_r + flag_r

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total requests", summary.get("total_requests", 0))
col2.metric(
    "Detection rate",
    f"{detection_rate * 100:.1f}%",
    help="block + sanitize + flag rate combined — the gateway's overall catch rate.",
)
col3.metric("Block rate", f"{block_r * 100:.1f}%")
col4.metric("Avg fusion latency", f"{float(summary.get('avg_total_latency_ms') or summary.get('avg_latency_ms') or 0):.0f} ms")
col5.metric(
    "Health",
    health.get("status", "?"),
    delta=f"{health.get('passed', 0)}/{health.get('total', 0)} checks",
)

# ---------------------------------------------------------------------------
# Module breakdown (avg latency, block contribution)
# ---------------------------------------------------------------------------
st.subheader("Per-module performance")
mod_rows = []
for mod in ("prompt_guard", "rag_guard", "output_agency", "output_guard"):
    avg_lat = summary.get(f"module_{mod}_avg_latency_ms")
    blocks = summary.get(f"module_{mod}_block_count")
    if avg_lat is None and blocks is None:
        continue
    mod_rows.append({
        "module": mod,
        "avg_latency_ms": avg_lat,
        "block_count": blocks,
    })

if mod_rows:
    st.dataframe(mod_rows, width="stretch", hide_index=True)
else:
    st.info("No per-module telemetry yet — run a few requests through the gateway.")


# ---------------------------------------------------------------------------
# Recent decisions
# ---------------------------------------------------------------------------
st.subheader("Recent decisions")
try:
    recent = client.get_json("/dashboard/recent-runs", params={"limit": 10}) or {}
except GatewayError as exc:
    st.warning(f"Could not load recent runs: {exc}")
    recent = {}

items = recent.get("decisions") or []
if items:
    st.dataframe(items, width="stretch", hide_index=True)
else:
    st.info("No fusion decisions logged yet.")
