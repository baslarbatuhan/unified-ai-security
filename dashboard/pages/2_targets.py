"""Targets — registered chatbots for external evaluation (CRUD)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from dashboard.lib.gateway_client import GatewayError, get_default_client


st.set_page_config(page_title="Targets", page_icon=":dart:", layout="wide")
st.title("Targets")
st.caption(
    "Chatbots registered in `external_eval/targets.yaml`. "
    "Add/edit through the form below — all writes go through the gateway "
    "with full Pydantic validation."
)
client = get_default_client()


def _load_targets(enabled_only: bool):
    payload = client.get_json("/targets", params={"enabled_only": str(enabled_only).lower()}) or {}
    return payload.get("targets") or []


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------
enabled_only = st.checkbox("Enabled only", value=False)
try:
    targets = _load_targets(enabled_only)
except GatewayError as exc:
    st.error(f"Gateway unreachable: {exc}")
    st.stop()

if targets:
    st.write(f"{len(targets)} target(s)")
    st.dataframe(
        [
            {
                "id": t.get("id"),
                "type": t.get("type"),
                "endpoint": t.get("endpoint", t.get("url", "")),
                "enabled": t.get("enabled", True),
                "has_tools": t.get("has_tools", False),
                "timeout_s": t.get("timeout_seconds"),
            }
            for t in targets
        ],
        width="stretch",
        hide_index=True,
    )
else:
    st.info("No targets configured yet.")

# ---------------------------------------------------------------------------
# Inspect + delete + "Edit ↓" (preloads the form below)
# ---------------------------------------------------------------------------
ids = [t["id"] for t in targets]
if ids:
    col_pick, col_edit, col_del = st.columns([3, 1, 1])
    chosen = col_pick.selectbox("Inspect target", ids)

    if col_edit.button("✏️ Edit ↓", width="stretch", help="Copy this target into the form below."):
        try:
            detail = client.get_json(f"/targets/{chosen}")
            # Store in session state; the form below reads _prefill to set its defaults.
            st.session_state["_prefill"] = detail
        except GatewayError as exc:
            st.error(f"Could not load: {exc}")

    if col_del.button("Delete", type="secondary", width="stretch"):
        try:
            client.delete(f"/targets/{chosen}")
        except GatewayError as exc:
            st.error(f"Delete failed: {exc}")
        else:
            st.session_state.pop("_prefill", None)
            st.success(f"Deleted target `{chosen}`")
            st.rerun()

    if chosen:
        try:
            detail = client.get_json(f"/targets/{chosen}")
            st.code(json.dumps(detail, indent=2), language="json")
        except GatewayError as exc:
            st.error(f"Could not load: {exc}")

# ---------------------------------------------------------------------------
# Add / upsert form — pre-filled when "Edit ↓" was clicked above.
# ---------------------------------------------------------------------------
st.markdown("---")
_pf = st.session_state.get("_prefill") or {}
_editing = bool(_pf)
st.subheader(f"{'Edit: ' + _pf['id'] if _editing else 'Add or update target'}")
if _editing:
    st.info(
        f"Editing **{_pf['id']}** — fields pre-filled from the saved record. "
        "Clear the id field or change it to create a new target instead."
    )

# Derive form defaults from prefill (or fall back to blank/safe values).
_type_opts = ["api", "web", "mock"]
_type_idx = _type_opts.index(_pf.get("type", "api")) if _pf.get("type") in _type_opts else 0
_auth_pf = _pf.get("auth") or {}
_auth_type_opts = ["none", "bearer", "api_key", "basic"]
_auth_type_val = _auth_pf.get("type", "none")
_auth_type_idx = _auth_type_opts.index(_auth_type_val) if _auth_type_val in _auth_type_opts else 0
_rt_default = (
    json.dumps(_pf["request_template"], indent=2)
    if _pf.get("request_template")
    else ""
)

with st.form("upsert_target"):
    cA, cB = st.columns(2)
    new_id = cA.text_input("id", value=_pf.get("id", ""), placeholder="internal_chatbot_api")
    target_type = cB.selectbox("type", _type_opts, index=_type_idx)
    name = st.text_input("name", value=_pf.get("name", ""), placeholder="Internal Chatbot (REST)")
    endpoint = st.text_input(
        "endpoint / URL",
        value=_pf.get("endpoint", ""),
        placeholder="https://chatbot.example/api/chat",
    )
    cC, cD, cE = st.columns(3)
    timeout_s = cC.number_input(
        "timeout (s)",
        min_value=1.0, max_value=120.0,
        value=float(_pf.get("timeout_seconds", 30.0)),
        step=1.0,
    )
    enabled = cD.checkbox("enabled", value=bool(_pf.get("enabled", True)))
    has_tools = cE.checkbox("has_tools", value=bool(_pf.get("has_tools", False)))

    auth_type = st.selectbox("auth.type", _auth_type_opts, index=_auth_type_idx)
    auth_token_env = st.text_input(
        "auth.token_env (env-var name; the actual value is never sent here)",
        value=_auth_pf.get("token_env", ""),
        placeholder="CHATBOT_BEARER_TOKEN",
        disabled=auth_type == "none",
    )

    st.markdown("**API adapter** *(api targets only)*")
    response_path = st.text_input(
        "response_path",
        value=_pf.get("response_path", ""),
        placeholder="choices.0.message.content",
        help="Dot-path into the JSON response that holds the assistant text.",
    )
    request_template_raw = st.text_area(
        "request_template (JSON)",
        value=_rt_default,
        placeholder='{"messages": [{"role": "user", "content": "{prompt}"}]}',
        help=(
            "Optional request body template. Use {prompt} as the placeholder "
            "for the attack text. Leave blank to use the default."
        ),
    )

    btn_label = f"Update {_pf['id']}" if _editing else "Save"
    submitted = st.form_submit_button(btn_label, width="stretch")

if not submitted:
    st.stop()

if not new_id or (target_type != "mock" and not endpoint):
    st.error("`id` and (for non-mock targets) `endpoint` are required.")
    st.stop()

payload = {
    "id": new_id,
    "name": name or new_id,
    "type": target_type,
    "enabled": enabled,
    "has_tools": has_tools,
    "timeout_seconds": float(timeout_s),
}
if endpoint:
    payload["endpoint"] = endpoint
if auth_type != "none":
    payload["auth"] = {"type": auth_type}
    if auth_token_env:
        payload["auth"]["token_env"] = auth_token_env
if response_path.strip():
    payload["response_path"] = response_path.strip()
if request_template_raw.strip():
    try:
        payload["request_template"] = json.loads(request_template_raw)
    except json.JSONDecodeError as _je:
        st.error(f"request_template is not valid JSON: {_je}")
        st.stop()

try:
    result = client.post_json("/targets", payload)
except GatewayError as exc:
    st.error(f"Save failed: {exc}")
else:
    st.session_state.pop("_prefill", None)   # Clear prefill after successful save.
    st.success(f"Saved target `{result.get('id', new_id)}`.")
    st.rerun()
