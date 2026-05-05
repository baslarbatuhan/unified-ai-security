"""Targets — registered chatbots for external evaluation (CRUD).

Form layout is *type-aware*: the `type` selectbox lives outside the
form so the visible field set updates the moment the user picks
`api` / `web` / `mock`. Inside the form, fields are gated on the
selected type so unrelated config never leaks into the save payload
(the backend's Pydantic validator would reject mixed shapes anyway,
but UI-side gating gives faster feedback).
"""
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
                "method": (t.get("http_method", "POST") if t.get("type") == "api" else "—"),
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

# ---------- TYPE + METHOD SELECTORS — live OUTSIDE the form ----------
# Streamlit `st.form` freezes widget values until the form is submitted,
# so any selectbox that gates which OTHER fields render must sit outside
# the form to trigger an immediate rerun. We need this for both the
# target `type` (api/web/mock) and the api `http_method` (POST/GET).
_type_opts = ["api", "web", "mock"]
_type_default = _pf.get("type") or "api"
_type_idx = _type_opts.index(_type_default) if _type_default in _type_opts else 0
target_type = st.selectbox(
    "type", _type_opts, index=_type_idx,
    help=(
        "api  = REST/HTTP endpoint (POST JSON body or GET query string). "
        "web  = Playwright-driven web chatbot. "
        "mock = local in-process echo, no network."
    ),
)

# `http_method` only meaningful for api targets; rendered next to `type`
# so the form below can reactively show request_template vs query_template.
http_method = "POST"
if target_type == "api":
    _method_opts = ["POST", "GET"]
    _method_default = _pf.get("http_method") or "POST"
    _method_idx = _method_opts.index(_method_default) if _method_default in _method_opts else 0
    http_method = st.selectbox(
        "HTTP method", _method_opts, index=_method_idx,
        help=(
            "POST → sends `request_template` as a JSON body. "
            "GET  → sends `query_template` as a URL-encoded query "
            "string. Pick GET when the chatbot uses an endpoint like "
            "`/chat?message=...` (curl `-G --data-urlencode`)."
        ),
    )

# Common form-time defaults derived from prefill regardless of type.
_auth_pf = _pf.get("auth") or {}
_auth_type_opts = ["none", "bearer", "header", "basic"]
_auth_type_val = _auth_pf.get("type") or "none"
_auth_type_idx = _auth_type_opts.index(_auth_type_val) if _auth_type_val in _auth_type_opts else 0

_rt_default = (
    json.dumps(_pf["request_template"], indent=2)
    if _pf.get("request_template")
    else ""
)
_qt_default = (
    json.dumps(_pf["query_template"], indent=2)
    if _pf.get("query_template")
    else ""
)
_selectors_pf = _pf.get("selectors") or {}


with st.form("upsert_target"):
    # ---------------- Common fields (visible for every type) ----------------
    cA, cB = st.columns(2)
    new_id = cA.text_input("id", value=_pf.get("id", ""), placeholder="hoca_n4b")
    name = cB.text_input("name", value=_pf.get("name", ""), placeholder="Hoca N4B Chatbot")
    cC, cD, cE = st.columns(3)
    timeout_s = cC.number_input(
        "timeout (s)",
        min_value=1.0, max_value=120.0,
        value=float(_pf.get("timeout_seconds", 30.0)),
        step=1.0,
    )
    enabled = cD.checkbox("enabled", value=bool(_pf.get("enabled", True)))
    has_tools = cE.checkbox("has_tools", value=bool(_pf.get("has_tools", False)))

    # Per-type initialisation so post-submit code can reference these
    # variables unconditionally — even when the active type doesn't render
    # a particular widget.
    endpoint = ""
    request_template_raw = ""
    query_template_raw = ""
    response_path = ""
    sel_input = ""
    sel_response = ""
    sel_submit = ""
    response_wait_ms = 3000
    auth_type = "none"
    auth_token_env = ""

    if target_type == "api":
        st.markdown(f"### API target ({http_method})")
        endpoint = st.text_input(
            "endpoint",
            value=_pf.get("endpoint") or "",
            placeholder="http://10.147.10.22:9993/chat/n4bchatbot",
        )

        if http_method == "POST":
            request_template_raw = st.text_area(
                "request_template (JSON)",
                value=_rt_default,
                placeholder='{"messages": [{"role": "user", "content": "{prompt}"}]}',
                help=(
                    "Optional request body template. Use `{prompt}` placeholder "
                    "for the attack text. Leave blank to use the default."
                ),
            )
        else:  # GET
            query_template_raw = st.text_area(
                "query_template (JSON)",
                value=_qt_default,
                placeholder='{"message": "{prompt}"}',
                help=(
                    "Required for GET. Flat key→value JSON; values may "
                    "contain `{prompt}`. Sent as `?key=encoded-value`."
                ),
            )

        response_path = st.text_input(
            "response_path (optional)",
            value=_pf.get("response_path") or "",
            placeholder="choices.0.message.content",
            help=(
                "Dot-path into a JSON response. Leave blank for plain-text "
                "endpoints — the adapter will return the body as-is."
            ),
        )

        st.markdown("**Auth**")
        auth_type = st.selectbox("auth.type", _auth_type_opts, index=_auth_type_idx)
        auth_token_env = st.text_input(
            "auth.token_env (env-var name; the actual value is never sent here)",
            value=_auth_pf.get("token_env") or "",
            placeholder="CHATBOT_BEARER_TOKEN",
            disabled=auth_type == "none",
        )

    elif target_type == "web":
        st.markdown("### Web target (Playwright)")
        endpoint = st.text_input(
            "page URL",
            value=_pf.get("endpoint") or "",
            placeholder="https://chatbot.example/",
        )
        sel_input = st.text_input(
            "selectors.input  *(required)*",
            value=_selectors_pf.get("input") or "",
            placeholder="textarea[name='message']",
            help="CSS selector for the prompt input element.",
        )
        sel_response = st.text_input(
            "selectors.response  *(required)*",
            value=_selectors_pf.get("response") or "",
            placeholder=".chat-message:last-child .content",
            help="CSS selector for the assistant's latest reply.",
        )
        sel_submit = st.text_input(
            "selectors.submit  (optional — adapter presses Enter if blank)",
            value=_selectors_pf.get("submit") or "",
            placeholder="button[type='submit']",
        )
        response_wait_ms = st.number_input(
            "response_wait_ms",
            min_value=0, max_value=120_000,
            value=int(_selectors_pf.get("response_wait_ms") or 3000),
            step=500,
            help="Delay between submit and response read; covers token streaming.",
        )

        st.markdown("**Auth**")
        auth_type = st.selectbox("auth.type", _auth_type_opts, index=_auth_type_idx)
        auth_token_env = st.text_input(
            "auth.token_env",
            value=_auth_pf.get("token_env") or "",
            disabled=auth_type == "none",
        )

    elif target_type == "mock":
        st.markdown("### Mock target")
        st.caption(
            "Local in-process adapter. Returns a deterministic echo of "
            "the prompt — useful for offline development. No endpoint, "
            "no auth, no template configuration required."
        )

    # ---------- Test-connection probe + dual submit buttons ----------
    st.markdown("---")
    probe_prompt = st.text_input(
        "Probe prompt (used by 🔌 Test connection)",
        value="ping",
        help=(
            "Single short message sent to the target when you click "
            "**🔌 Test connection** below. Default `ping` is benign and "
            "won't trigger any guard. Hocanın chatbot'u Türkçe varsayılana "
            "uyacak şekilde yapılandırıldıysa `Merhaba` deneyebilirsin."
        ),
    )
    btn_label = f"Update {_pf['id']}" if _editing else "Save"
    col_save, col_test = st.columns([2, 1])
    submitted = col_save.form_submit_button(btn_label, width="stretch")
    test_clicked = col_test.form_submit_button(
        "🔌 Test connection",
        width="stretch",
        help="Send the probe prompt to this target without saving it.",
    )

if not submitted and not test_clicked:
    st.stop()

# ---------------------------------------------------------------------------
# Validation + payload assembly
# ---------------------------------------------------------------------------
if not new_id:
    st.error("`id` is required.")
    st.stop()

payload = {
    "id": new_id,
    "name": name or new_id,
    "type": target_type,
    "enabled": enabled,
    "has_tools": has_tools,
    "timeout_seconds": float(timeout_s),
}

if target_type == "api":
    if not endpoint:
        st.error("API target requires an `endpoint`.")
        st.stop()
    payload["endpoint"] = endpoint
    payload["http_method"] = http_method

    rt_text = (request_template_raw or "").strip()
    qt_text = (query_template_raw or "").strip()
    rp_text = (response_path or "").strip()

    if http_method == "POST" and rt_text:
        try:
            payload["request_template"] = json.loads(rt_text)
        except json.JSONDecodeError as exc:
            st.error(f"request_template is not valid JSON: {exc}")
            st.stop()
    if http_method == "GET":
        if not qt_text:
            st.error(
                "GET-style API target requires `query_template` "
                '(e.g. `{"message": "{prompt}"}`).'
            )
            st.stop()
        try:
            payload["query_template"] = json.loads(qt_text)
        except json.JSONDecodeError as exc:
            st.error(f"query_template is not valid JSON: {exc}")
            st.stop()
    if rp_text:
        payload["response_path"] = rp_text

    if auth_type != "none":
        payload["auth"] = {"type": auth_type}
        if auth_token_env:
            payload["auth"]["token_env"] = auth_token_env

elif target_type == "web":
    if not endpoint:
        st.error("Web target requires a page URL in `endpoint`.")
        st.stop()
    if not sel_input or not sel_response:
        st.error(
            "Web target requires both `selectors.input` and "
            "`selectors.response`."
        )
        st.stop()
    payload["endpoint"] = endpoint
    selectors = {
        "input": sel_input,
        "response": sel_response,
        "response_wait_ms": int(response_wait_ms),
    }
    submit_sel = (sel_submit or "").strip()
    if submit_sel:
        selectors["submit"] = submit_sel
    payload["selectors"] = selectors

    if auth_type != "none":
        payload["auth"] = {"type": auth_type}
        if auth_token_env:
            payload["auth"]["token_env"] = auth_token_env

# mock: only common fields are sent — the backend default for empty
# auth/endpoint/template covers it.

if test_clicked:
    # Dry-run probe — full target dict to /targets/test, no YAML write.
    with st.spinner(f"Testing connection to {payload.get('endpoint') or new_id}…"):
        try:
            result = client.post_json(
                "/targets/test",
                {"target": payload, "probe_prompt": probe_prompt or "ping"},
            )
        except GatewayError as exc:
            st.error(f"Probe request failed: {exc}")
            st.stop()

    if result.get("ok"):
        lat = result.get("latency_ms", 0)
        meta = result.get("metadata") or {}
        sample = result.get("response_sample") or "(empty response)"
        st.success(
            f"✅ Connected — {lat} ms · "
            f"status={meta.get('status_code', '—')} · "
            f"content-type={meta.get('content_type', '—')} · "
            f"{result.get('response_chars', 0)} chars"
        )
        with st.expander("Response sample (first 200 chars)", expanded=True):
            st.code(sample)
        with st.expander("Raw metadata", expanded=False):
            st.json(meta)
    else:
        # Categorise → colour-coded banner.
        cat = result.get("category", "unexpected")
        emoji = {
            "timeout":   "🟠",
            "transport": "🔴",
            "schema":    "🟡",
            "config":    "🟣",
            "unexpected": "⚫",
        }.get(cat, "❌")
        st.error(
            f"{emoji} Failed ({cat or 'unknown'}) — "
            f"{result.get('error_message', 'no message')}"
        )
        if result.get("error_details"):
            with st.expander("Schema error details", expanded=False):
                st.json(result["error_details"])
        if result.get("metadata"):
            with st.expander("Adapter metadata", expanded=False):
                st.json(result["metadata"])
    st.stop()

# --- Save path ---
try:
    result = client.post_json("/targets", payload)
except GatewayError as exc:
    st.error(f"Save failed: {exc}")
else:
    st.session_state.pop("_prefill", None)   # Clear prefill after successful save.
    st.success(f"Saved target `{result.get('id', new_id)}`.")
    st.rerun()
