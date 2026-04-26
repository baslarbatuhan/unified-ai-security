"""dashboard/app.py
Streamlit entry point for the Unified AI Security Gateway dashboard.

Streamlit auto-discovers files under ``dashboard/pages/`` and renders
them in the sidebar in alphanumeric order — that is why each page file
is prefixed with a number (``1_home.py``, ``2_targets.py``, …).

The dashboard talks to the gateway over HTTP. The base URL defaults to
``http://127.0.0.1:8000`` and can be overridden with the
``GATEWAY_URL`` environment variable so the same UI can point at a
remote deployment without code changes.

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

# Streamlit invokes each script in isolation; project root isn't on
# sys.path by default, so dashboard.* imports fail. Inject it before
# the first project import.
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from dashboard.lib.gateway_client import GatewayClient, get_default_client


def main() -> None:
    st.set_page_config(
        page_title="Unified AI Security Gateway",
        page_icon=":lock:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    client: GatewayClient = get_default_client()

    st.title("Unified AI Security Gateway")
    st.caption(
        "Operator dashboard — pick a page from the sidebar. "
        f"Connected to gateway at `{client.base_url}`."
    )

    st.markdown(
        """
        **Pages**

        - **Home** — system snapshot: recent decisions, success rate, latency, health.
        - **Targets** — chatbots registered for external evaluation.
        - **Run test** — pick suite/modules/threshold and launch an evaluation.
        - **Live monitor** — real-time telemetry tail.
        - **Results** — attack matrix, P/R/F1, filterable per-class.
        - **Reports** — generate, view, and download markdown reports.
        - **Logs** — explainability viewer (which signal fired, which chunk).

        Every test run is config-driven. The exact configuration is snapshotted
        to ``runs/<run_id>/config_used.yaml`` so any number can be reproduced.
        """
    )

    with st.sidebar:
        st.header("Gateway")
        st.code(client.base_url, language="text")
        if st.button("Ping /health", width="stretch"):
            try:
                health = client.get_json("/health")
                status = health.get("status", "unknown")
                if status == "OK":
                    st.success(f"healthy ({health.get('passed', 0)}/{health.get('total', 0)})")
                else:
                    st.warning(f"status={status}")
            except Exception as exc:  # noqa: BLE001 — surface any connectivity error
                st.error(f"unreachable: {exc}")


if __name__ == "__main__":
    main()
