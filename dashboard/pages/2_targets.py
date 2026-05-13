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


def _parse_json_object(label: str, raw: str) -> dict | None:
    """Parse a JSON object from a form text-area. Empty → empty dict.

    Surfaces a user-visible st.error and returns sentinel `None` on
    parse failure so the caller can short-circuit the save. Non-object
    JSON (lists, scalars) is rejected — headers must be a flat object.
    """
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        st.error(f"{label} is not valid JSON: {exc}")
        return None
    if not isinstance(parsed, dict):
        st.error(f"{label} must be a JSON object, got {type(parsed).__name__}")
        return None
    # Coerce values to strings — Pydantic AuthHeader.headers is Dict[str, str].
    return {str(k): str(v) for k, v in parsed.items()}


def _assemble_auth_payload(
    *,
    auth_type: str,
    token_env: str, token_inline: str,
    headers_raw: str,
    query_key: str,
    query_value_env: str, query_value_inline: str,
    basic_user: str, basic_user_env: str,
    basic_pass_env: str,
    extra_headers_raw: str,
) -> dict | None:
    """Build the `auth` dict to send in the POST/PATCH payload.

    Returns `None` when the form is in an invalid state (a st.error has
    already been emitted) — caller should st.stop() in that case.
    Returns `{}` to signal "send no auth field" (server applies AuthNone).
    """
    extra_headers = _parse_json_object("auth.extra_headers", extra_headers_raw)
    if extra_headers is None:
        return None

    if auth_type == "none":
        # Always send `auth.type=none` so the backend round-trips cleanly
        # even when extra_headers is non-empty.
        return {"type": "none", "extra_headers": extra_headers}

    if auth_type == "bearer":
        if not (token_env or token_inline):
            st.error("bearer auth: provide either `auth.token_env` or `auth.token`.")
            return None
        body: dict = {"type": "bearer", "extra_headers": extra_headers}
        if token_env:
            body["token_env"] = token_env
        if token_inline:
            body["token"] = token_inline
        return body

    if auth_type == "header":
        headers = _parse_json_object("auth.headers", headers_raw)
        if headers is None:
            return None
        if not headers:
            st.error("header auth: `auth.headers` cannot be empty.")
            return None
        return {"type": "header", "headers": headers, "extra_headers": extra_headers}

    if auth_type == "query":
        if not query_key:
            st.error("query auth: `auth.query_key` is required (e.g. `key`).")
            return None
        if not (query_value_env or query_value_inline):
            st.error("query auth: provide `auth.query_value_env` or `auth.query_value`.")
            return None
        body = {"type": "query", "query_key": query_key, "extra_headers": extra_headers}
        if query_value_env:
            body["query_value_env"] = query_value_env
        if query_value_inline:
            body["query_value"] = query_value_inline
        return body

    if auth_type == "basic":
        if not (basic_user or basic_user_env):
            st.error("basic auth: provide `auth.username` or `auth.username_env`.")
            return None
        if not basic_pass_env:
            st.error("basic auth: `auth.password_env` is required (no inline password input).")
            return None
        body = {"type": "basic", "extra_headers": extra_headers}
        if basic_user:
            body["username"] = basic_user
        if basic_user_env:
            body["username_env"] = basic_user_env
        body["password_env"] = basic_pass_env
        return body

    # Unknown / future type — surface and bail.
    st.error(f"unsupported auth.type {auth_type!r}")
    return None


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
# Hafta 11.2: 5-tip discriminated union. `auth_type` selectbox lives
# OUTSIDE the form so changing it re-renders the matching widget group
# immediately (form widgets freeze until submit).
_auth_type_opts = ["none", "bearer", "header", "query", "basic"]
_auth_type_val = _auth_pf.get("type") or "none"
_auth_type_idx = _auth_type_opts.index(_auth_type_val) if _auth_type_val in _auth_type_opts else 0
# Render the auth-type selector only for target types that consume auth
# (api / web). Mock targets ignore it. The variable is referenced
# unconditionally further down, so initialise to "none" first.
auth_type = "none"
if target_type in ("api", "web"):
    auth_type = st.selectbox(
        "auth.type",
        _auth_type_opts,
        index=_auth_type_idx,
        help=(
            "none   = open endpoint, no credentials. "
            "bearer = Authorization: Bearer <token> (e.g. OpenAI, internal JWT). "
            "header = arbitrary header(s) (e.g. X-API-Key). "
            "query  = `?key=...` query-param auth (e.g. Gemini). "
            "basic  = HTTP Basic auth."
        ),
    )

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
    # Hafta 11.2 auth-related widget defaults. Populated below depending on
    # the (outside-form) `auth_type` value. `auth_type` itself is already
    # set above; what we collect here are the per-variant inputs.
    auth_token_env = ""
    auth_token_inline = ""
    auth_headers_raw = ""
    auth_query_key = ""
    auth_query_value_env = ""
    auth_query_value_inline = ""
    auth_basic_user = ""
    auth_basic_user_env = ""
    auth_basic_pass_env = ""
    auth_extra_headers_raw = ""

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

        # Hafta 11.2: 5-type auth block. `auth_type` already selected
        # outside the form; render only the matching variant's inputs +
        # the shared extra_headers editor.
        st.markdown(f"**Auth** (`type={auth_type}`)")
        if auth_type == "bearer":
            auth_token_env = st.text_input(
                "auth.token_env  (env-var name; **the secret never enters the form**)",
                value=_auth_pf.get("token_env") or "",
                placeholder="OPENAI_API_KEY",
                help="Adapter reads os.environ[<this>] at request time.",
            )
            auth_token_inline = st.text_input(
                "auth.token  (legacy / tests — prefer token_env)",
                value="",  # never prefill; redacted on read
                placeholder="leave blank if using token_env",
                type="password",
            )
        elif auth_type == "header":
            _pf_headers = _auth_pf.get("headers") or {}
            auth_headers_raw = st.text_area(
                "auth.headers  (JSON object — key→value)",
                value=json.dumps(_pf_headers, indent=2) if _pf_headers else "",
                placeholder='{"X-API-Key": "${MY_KEY_ENV}"}',
                help="Static headers added on every request. Use ${ENV} to interpolate at runtime (future enhancement; for now hard-code or use env-substitution at YAML load).",
            )
        elif auth_type == "query":
            auth_query_key = st.text_input(
                "auth.query_key",
                value=_auth_pf.get("query_key") or "",
                placeholder="key",
                help="Query param name appended to every request URL (e.g. `key` for Gemini).",
            )
            auth_query_value_env = st.text_input(
                "auth.query_value_env",
                value=_auth_pf.get("query_value_env") or "",
                placeholder="GEMINI_API_KEY",
                help="Env-var holding the secret value.",
            )
            auth_query_value_inline = st.text_input(
                "auth.query_value  (legacy / tests — prefer query_value_env)",
                value="",
                placeholder="leave blank if using query_value_env",
                type="password",
            )
        elif auth_type == "basic":
            cBA, cBB = st.columns(2)
            auth_basic_user = cBA.text_input(
                "auth.username  (inline)",
                value=_auth_pf.get("username") or "",
            )
            auth_basic_user_env = cBB.text_input(
                "auth.username_env",
                value=_auth_pf.get("username_env") or "",
            )
            auth_basic_pass_env = st.text_input(
                "auth.password_env  (no inline password input — use an env-var)",
                value=_auth_pf.get("password_env") or "",
                placeholder="MY_BASIC_PASSWORD_ENV",
            )

        # Shared across all types: extra static headers (OpenAI org id,
        # tenant id, Cloudflare access headers, …). Visible even when
        # auth.type=none so open endpoints can still pin custom headers.
        _pf_extra = _auth_pf.get("extra_headers") or {}
        auth_extra_headers_raw = st.text_area(
            "auth.extra_headers  (JSON object — optional, attached on every request)",
            value=json.dumps(_pf_extra, indent=2) if _pf_extra else "",
            placeholder='{"OpenAI-Organization": "org-xyz"}',
            help=(
                "Static headers merged onto every request regardless of "
                "auth.type. Use for vendor-specific identifiers like "
                "OpenAI-Organization, X-Tenant-Id, Accept-Version pinning."
            ),
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

        # Web targets honour auth.extra_headers (e.g. Cloudflare Access)
        # but the Playwright adapter doesn't apply bearer/header/query
        # itself. Surface the extra_headers field; the rest is informational.
        st.markdown(f"**Auth** (`type={auth_type}`)")
        if auth_type != "none":
            st.caption(
                "ℹ️ Web (Playwright) adapter currently applies only "
                "`extra_headers`. bearer / header / query / basic auth is "
                "API-only for now."
            )
        _pf_extra = _auth_pf.get("extra_headers") or {}
        auth_extra_headers_raw = st.text_area(
            "auth.extra_headers  (JSON object — optional)",
            value=json.dumps(_pf_extra, indent=2) if _pf_extra else "",
            placeholder='{"Accept-Language": "en-US"}',
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

    # Hafta 11.2: build auth payload from the variant's inputs. The
    # discriminated union validator will reject incomplete shapes
    # (e.g. bearer with no token sources, query with no value).
    auth_payload = _assemble_auth_payload(
        auth_type=auth_type,
        token_env=auth_token_env, token_inline=auth_token_inline,
        headers_raw=auth_headers_raw,
        query_key=auth_query_key,
        query_value_env=auth_query_value_env, query_value_inline=auth_query_value_inline,
        basic_user=auth_basic_user, basic_user_env=auth_basic_user_env,
        basic_pass_env=auth_basic_pass_env,
        extra_headers_raw=auth_extra_headers_raw,
    )
    if auth_payload is not None:
        payload["auth"] = auth_payload

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

    # Hafta 11.2: build auth payload from the variant's inputs. The
    # discriminated union validator will reject incomplete shapes
    # (e.g. bearer with no token sources, query with no value).
    auth_payload = _assemble_auth_payload(
        auth_type=auth_type,
        token_env=auth_token_env, token_inline=auth_token_inline,
        headers_raw=auth_headers_raw,
        query_key=auth_query_key,
        query_value_env=auth_query_value_env, query_value_inline=auth_query_value_inline,
        basic_user=auth_basic_user, basic_user_env=auth_basic_user_env,
        basic_pass_env=auth_basic_pass_env,
        extra_headers_raw=auth_extra_headers_raw,
    )
    if auth_payload is not None:
        payload["auth"] = auth_payload

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
