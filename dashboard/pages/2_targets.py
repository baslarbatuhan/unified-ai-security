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
from dashboard.lib.target_presets import get_preset, preset_names


st.set_page_config(page_title="Targets", page_icon=":dart:", layout="wide")
st.title("Targets")
st.caption(
    "Chatbots registered in `external_eval/targets.yaml`. "
    "Add/edit through the form below — all writes go through the gateway "
    "with full Pydantic validation."
)
client = get_default_client()


def _fetch_vault_state() -> dict:
    """Map ``env_name -> source`` ('env' / 'vault' / 'missing').

    Read once per page render so each ``*_env`` field can render a small
    badge ("🔒 stored" / "🌱 from env-var" / "⚠ not set") without an
    extra round-trip per input. Failures degrade silently to an empty
    map; the form is still usable, badges just disappear.
    """
    try:
        data = client.get_json("/secrets") or {}
    except GatewayError:
        return {}
    items = data.get("items") or []
    return {it.get("name", ""): it.get("source", "missing") for it in items if it.get("name")}


def _vault_badge(env_name: str, vault_state: dict) -> str:
    """Markdown caption text for the badge under an ``*_env`` input."""
    if not env_name:
        return ""
    src = vault_state.get(env_name, "missing")
    return {
        "env": f"🌱 `{env_name}` resolved from a container env-var (vault is shadowed)",
        "vault": f"🔒 `{env_name}` stored in local vault — ready to use",
        "missing": f"⚠️ `{env_name}` has no value yet — paste it below or set the env-var",
    }.get(src, "")


def _push_vault_values(pairs: list) -> list:
    """For each ``(env_name, value)`` pair where both are non-empty,
    PUT the value into the gateway's secrets vault. Returns a list of
    user-visible status strings (success or failure per push)."""
    statuses: list = []
    for env_name, value in pairs:
        env_name = (env_name or "").strip()
        value = value or ""
        if not env_name or not value:
            continue
        try:
            client.put_json(f"/secrets/{env_name}", {"value": value})
            statuses.append(f"🔒 Stored `{env_name}` in local vault.")
        except GatewayError as exc:
            statuses.append(f"❌ Could not store `{env_name}`: {exc}")
    return statuses


vault_state = _fetch_vault_state()


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
# Presets fill endpoint/auth/template only — no ``id``. Treat as "edit"
# only when a saved target was loaded via "Edit ↓" (always has ``id``).
_editing = bool(_pf.get("id"))
st.subheader(f"{'Edit: ' + _pf['id'] if _editing else 'Add or update target'}")
if _editing:
    st.info(
        f"Editing **{_pf['id']}** — fields pre-filled from the saved record. "
        "Clear the id field or change it to create a new target instead."
    )
elif _pf:
    st.info(
        "Preset applied — fill in **id** (and name if you like), paste secrets "
        "into the vault fields, then **Save**."
    )

# ---------- Sprint 6: PROVIDER PRESETS — outside the form, applied via session_state ----------
# A preset is a partial target dict (endpoint + auth shape + request
# template + response_path); picking one mutates `_prefill` so the form
# below reruns with the preset's values populated. The user still has
# to name the secret + paste it into the 🔐 vault field. Skipped while
# editing an existing target — that path is for tweaking, not cloning.
if not _editing:
    _preset_options = preset_names()
    _preset_choice = st.selectbox(
        "Preset (optional)",
        _preset_options,
        index=0,  # always defaults to "(custom)" so the form doesn't auto-populate
        help=(
            "Pre-fills endpoint, auth shape, request template and "
            "response_path for the named provider. Pick `(custom)` to "
            "build a target from scratch."
        ),
    )
    if _preset_choice and _preset_choice != "(custom)":
        if st.button(f"Apply preset: {_preset_choice}", key="_apply_preset"):
            preset = get_preset(_preset_choice)
            preset.setdefault("name", _preset_choice)
            st.session_state["_prefill"] = preset
            st.rerun()

# ---------- TYPE + METHOD SELECTORS — live OUTSIDE the form ----------
# Streamlit `st.form` freezes widget values until the form is submitted,
# so any selectbox that gates which OTHER fields render must sit outside
# the form to trigger an immediate rerun. We need this for both the
# target `type` (api/web/mock) and the api `http_method` (POST/GET).
_type_opts = ["api", "web", "mock", "tools_local"]
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
# Render the auth-type selector only for API targets. Web (Playwright)
# never applies bearer/header/query/basic — its adapter only honours
# `extra_headers` — so we surface only that field inside the web block
# and hide the misleading dropdown. Mock targets ignore auth entirely.
# Sprint 2: an existing web target carrying a non-none auth.type from
# a previous form revision is silently coerced to "none" on save (no
# input is rendered for the variant, so its fields default to empty).
auth_type = "none"
if target_type == "api":
    auth_type = st.selectbox(
        "auth.type",
        _auth_type_opts,
        index=_auth_type_idx,
        help=(
            "none   = open endpoint, no credentials. "
            "bearer = Authorization: Bearer <token> (e.g. OpenAI, internal JWT). "
            "header = arbitrary header(s) — supports ${VAR} interpolation. "
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
    has_tools = cE.checkbox(
        "has_tools",
        value=bool(_pf.get("has_tools", False)),
        help=(
            "Tick when the target exposes tool/function calling. The "
            "`agency_social` suite (and the `all`-mode equivalent of it) "
            "drops cases tagged `requires_tools=True` for targets where "
            "this is unchecked — leaving it off saves you from a run "
            "that filters every scenario out."
        ),
    )

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
    sel_fallback_input_raw = ""
    sel_fallback_response_raw = ""
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
    # Vault paste inputs — paired with the *_env fields above. If the
    # user types both a name and a value, the save path PUTs the value
    # into /secrets/{name} so the adapter can resolve it later without
    # touching .env or restarting the container.
    auth_token_vault_value = ""
    auth_query_vault_value = ""
    auth_basic_user_vault_value = ""
    auth_basic_pass_vault_value = ""
    # Sprint 1: header auth picks up secrets via ``${VAR}`` placeholders
    # inside ``auth.headers``. The user names the var and pastes the
    # value here; the save path PUTs it into /secrets/{name} so the
    # adapter resolves it at request time. (For multi-secret header
    # setups, save once per secret — the vault accumulates entries.)
    auth_header_secret_name = ""
    auth_header_secret_value = ""
    # Sprint 4: web cookie / storage_state auth.
    auth_cookies_raw = ""
    auth_storage_state_path = ""
    auth_cookie_secret_name = ""
    auth_cookie_secret_value = ""

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
                "auth.token_env  (env-var name — you choose it)",
                value=_auth_pf.get("token_env") or "",
                placeholder="MY_API_KEY",
                help=(
                    "Pick any name you like for this secret. The adapter "
                    "reads it from os.environ at request time, falling back "
                    "to the local secrets vault (set via this page) if the "
                    "env-var isn't defined."
                ),
            )
            _badge = _vault_badge(auth_token_env, vault_state)
            if _badge:
                st.caption(_badge)
            auth_token_vault_value = st.text_input(
                "🔐 Vault value for the env-var above (optional)",
                value="",
                type="password",
                placeholder="paste the API key here to save it in the local vault",
                help=(
                    "Stored under the env-var name above, in the git-ignored "
                    "vault at `runs/.secrets.yaml`. Leave blank to keep "
                    "whatever is already in the vault (or in the container's "
                    "real env-var)."
                ),
            )
            with st.expander("Advanced — inline token (legacy, writes to targets.yaml)", expanded=False):
                st.caption(
                    "⚠️ Embeds the secret directly in `targets.yaml`, which "
                    "is checked into git. Prefer the 🔐 vault field above. "
                    "Kept here for tests and offline edge cases only."
                )
                auth_token_inline = st.text_input(
                    "auth.token (inline)",
                    value="",  # never prefill; redacted on read
                    placeholder="leave blank — use the vault paste field instead",
                    type="password",
                )
        elif auth_type == "header":
            _pf_headers = _auth_pf.get("headers") or {}
            auth_headers_raw = st.text_area(
                "auth.headers  (JSON object — key→value)",
                value=json.dumps(_pf_headers, indent=2) if _pf_headers else "",
                placeholder='{"x-api-key": "${MY_API_KEY}"}',
                help=(
                    "Static headers added on every request. Reference a "
                    "secret with ``${VAR_NAME}`` — the adapter resolves "
                    "it at request time from the env-var first, then the "
                    "local vault. Literal values (without ``${}``) are "
                    "sent unchanged, so static identifiers like "
                    "``X-Tenant-Id`` work too."
                ),
            )
            # Vault paste pair — covers the common single-secret case
            # (Anthropic ``x-api-key``, Gemini ``x-goog-api-key``, custom
            # ``X-Auth-Token``…). For multi-secret headers, save once per
            # secret; the vault accumulates entries and never returns
            # values to the dashboard.
            auth_header_secret_name = st.text_input(
                "Secret name (the `${VAR}` you referenced above)",
                value="",
                placeholder="MY_API_KEY",
                help=(
                    "Type the env-var name without the ``${}`` wrapping. "
                    "Must match what you used inside the headers JSON."
                ),
            )
            _badge = _vault_badge(auth_header_secret_name, vault_state)
            if _badge:
                st.caption(_badge)
            auth_header_secret_value = st.text_input(
                "🔐 Vault value for the secret name above (optional)",
                value="",
                type="password",
                placeholder="paste the secret here to save it in the local vault",
                help=(
                    "Stored under the name above, in the git-ignored "
                    "vault at `runs/.secrets.yaml`. Leave blank to keep "
                    "whatever's already in the vault (or the container "
                    "env-var of the same name)."
                ),
            )
        elif auth_type == "query":
            auth_query_key = st.text_input(
                "auth.query_key",
                value=_auth_pf.get("query_key") or "",
                placeholder="key",
                help="Query param name appended to every request URL (e.g. `key` for Gemini).",
            )
            auth_query_value_env = st.text_input(
                "auth.query_value_env  (env-var name — you choose it)",
                value=_auth_pf.get("query_value_env") or "",
                placeholder="MY_API_KEY",
                help=(
                    "Pick any name you like for this secret. Resolved from "
                    "os.environ first, then the local secrets vault."
                ),
            )
            _badge = _vault_badge(auth_query_value_env, vault_state)
            if _badge:
                st.caption(_badge)
            auth_query_vault_value = st.text_input(
                "🔐 Vault value for the env-var above (optional)",
                value="",
                type="password",
                placeholder="paste the API key here to save it in the local vault",
                help=(
                    "Stored under the env-var name above, in the git-ignored "
                    "vault. Leave blank to keep whatever's already there."
                ),
            )
            with st.expander("Advanced — inline value (legacy, writes to targets.yaml)", expanded=False):
                st.caption(
                    "⚠️ Embeds the secret directly in `targets.yaml`, which "
                    "is checked into git. Prefer the 🔐 vault field above."
                )
                auth_query_value_inline = st.text_input(
                    "auth.query_value (inline)",
                    value="",
                    placeholder="leave blank — use the vault paste field instead",
                    type="password",
                )
        elif auth_type == "basic":
            cBA, cBB = st.columns(2)
            auth_basic_user = cBA.text_input(
                "auth.username  (inline)",
                value=_auth_pf.get("username") or "",
            )
            auth_basic_user_env = cBB.text_input(
                "auth.username_env  (env-var name — you choose it)",
                value=_auth_pf.get("username_env") or "",
                placeholder="MY_BASIC_USER",
            )
            _badge_u = _vault_badge(auth_basic_user_env, vault_state)
            if _badge_u:
                st.caption(_badge_u)
            auth_basic_user_vault_value = st.text_input(
                "🔐 Vault value for username_env (optional)",
                value="",
                type="password",
                placeholder="paste the username here to save it in the vault",
            )
            auth_basic_pass_env = st.text_input(
                "auth.password_env  (env-var name — you choose it)",
                value=_auth_pf.get("password_env") or "",
                placeholder="MY_BASIC_PASSWORD",
            )
            _badge_p = _vault_badge(auth_basic_pass_env, vault_state)
            if _badge_p:
                st.caption(_badge_p)
            auth_basic_pass_vault_value = st.text_input(
                "🔐 Vault value for password_env (optional)",
                value="",
                type="password",
                placeholder="paste the password here to save it in the vault",
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

        # ---------- Sprint 5.1: fallback selectors ----------
        # The WebAdapter tries primary selectors first, then walks each
        # fallback list in order — survives minor UI revs without code
        # changes. One selector per line; blank lines ignored. Empty
        # lists round-trip as omitted fields (no spurious yaml diff).
        with st.expander("Fallback selectors (optional — survives UI revs)", expanded=False):
            sel_fallback_input_raw = st.text_area(
                "fallback_input  (one CSS selector per line)",
                value="\n".join(_selectors_pf.get("fallback_input") or []),
                placeholder="textarea.chat-input\n#chat-textarea",
                help=(
                    "Alternates tried in order if `selectors.input` "
                    "stops matching after a UI revision."
                ),
                height=80,
            )
            sel_fallback_response_raw = st.text_area(
                "fallback_response  (one CSS selector per line)",
                value="\n".join(_selectors_pf.get("fallback_response") or []),
                placeholder=".message.assistant:last-child\n[data-role=assistant]",
                help=(
                    "Alternates tried in order if `selectors.response` "
                    "stops matching."
                ),
                height=80,
            )

        # ---------- Sprint 4: session auth (cookies / storage_state) ----------
        st.markdown("**Session auth** (optional — for CSRF / logged-in chatbots)")
        st.caption(
            "Leave both blank for open / public pages. Otherwise paste "
            "cookies copied from devtools, or point at a Playwright "
            "`storage_state.json` exported by `context.storage_state(path=...)`."
        )
        _pf_is_cookie = (_auth_pf.get("type") == "cookie")
        _pf_cookies = _auth_pf.get("cookies") if _pf_is_cookie else None
        _pf_storage = _auth_pf.get("storage_state_path") if _pf_is_cookie else None
        auth_storage_state_path = st.text_input(
            "storage_state_path  (optional)",
            value=_pf_storage or "",
            placeholder="/app/runs/storage_state_example.json",
            help=(
                "Container path to a Playwright storage_state JSON. "
                "Captures cookies + localStorage in one shot."
            ),
        )
        auth_cookies_raw = st.text_area(
            "cookies  (JSON list — optional; format matches devtools export)",
            value=json.dumps(_pf_cookies, indent=2) if _pf_cookies else "",
            placeholder=(
                '[\n'
                '  {"name": "session", "value": "${MY_SESSION_TOKEN}",\n'
                '   "domain": ".example.com", "path": "/"}\n'
                ']'
            ),
            help=(
                "List of cookies injected before the first page load. "
                "Values may reference `${VAR}` so the secret stays in "
                "the vault. `domain` / `url` are optional — falls back "
                "to the page URL."
            ),
            height=140,
        )
        auth_cookie_secret_name = st.text_input(
            "Cookie secret name (the `${VAR}` you referenced in the cookies above)",
            value="",
            placeholder="MY_SESSION_TOKEN",
            help="The env-var name you want this paste to land under in the vault.",
        )
        _badge = _vault_badge(auth_cookie_secret_name, vault_state)
        if _badge:
            st.caption(_badge)
        auth_cookie_secret_value = st.text_input(
            "🔐 Vault value for the cookie secret above (optional)",
            value="",
            type="password",
            placeholder="paste the session cookie value here to save it locally",
        )

        st.markdown("**Extra headers** (sent on every browser request)")
        _pf_extra = _auth_pf.get("extra_headers") or {}
        auth_extra_headers_raw = st.text_area(
            "auth.extra_headers  (JSON object — optional)",
            value=json.dumps(_pf_extra, indent=2) if _pf_extra else "",
            placeholder='{"Accept-Language": "en-US"}',
            help=(
                "Static headers attached on every Playwright request. "
                "Useful for Cloudflare Access (CF-Access-Client-Id) or "
                "tenant identifiers."
            ),
        )

    elif target_type == "mock":
        st.markdown("### Mock target")
        st.caption(
            "Local in-process adapter. Returns a deterministic echo of "
            "the prompt — useful for offline development. No endpoint, "
            "no auth, no template configuration required."
        )

    elif target_type == "tools_local":
        # Sprint 5.2: surface tools_local as a first-class type so users
        # can register the gateway-side tool sandbox (weather / stock /
        # calc — Hafta 14) without dropping into raw yaml. No endpoint,
        # no auth, no request template; the adapter runs registered
        # Python tools in-process against ATK-031..050 scenarios.
        st.markdown("### Tools (local sandbox)")
        st.caption(
            "Runs registered local tools (weather_forecast / stock_quote / "
            "calc_evaluate) in-process. No endpoint required. **Tick "
            "`has_tools` above** so agency_social scenarios are included "
            "in the suite."
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

# Push any vault values the user pasted into the form BEFORE the /targets
# call. That way both Test Connection and Save operate against an
# up-to-date vault — the adapter will resolve auth_token_env to the new
# value on the very next request. We do this even on Test Connection so
# the probe can succeed using a freshly pasted key.
for _msg in _push_vault_values([
    (auth_token_env, auth_token_vault_value),
    (auth_query_value_env, auth_query_vault_value),
    (auth_basic_user_env, auth_basic_user_vault_value),
    (auth_basic_pass_env, auth_basic_pass_vault_value),
    # Sprint 1: header auth uses a user-named secret referenced from
    # the headers JSON as ``${NAME}``. Push the pasted value before
    # the /targets call so Test connection sees the fresh secret.
    (auth_header_secret_name, auth_header_secret_value),
    # Sprint 4: same idea, for web cookie ${VAR} references.
    (auth_cookie_secret_name, auth_cookie_secret_value),
]):
    if _msg.startswith("❌"):
        st.error(_msg)
    else:
        st.success(_msg)

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
    # Sprint 5.1: parse fallback selector text-areas (one selector per
    # line; blank lines ignored).  Empty lists are omitted entirely so
    # `targets.yaml` stays clean for the common no-fallback case.
    fb_in = [ln.strip() for ln in (sel_fallback_input_raw or "").splitlines() if ln.strip()]
    fb_out = [ln.strip() for ln in (sel_fallback_response_raw or "").splitlines() if ln.strip()]
    if fb_in:
        selectors["fallback_input"] = fb_in
    if fb_out:
        selectors["fallback_response"] = fb_out
    payload["selectors"] = selectors

    # Sprint 4: decide between cookie auth (cookies / storage_state) and
    # auth.type=none (still honours extra_headers). The two web-only
    # fields drive the choice; if both are blank we stay on `none`.
    extra_headers = _parse_json_object("auth.extra_headers", auth_extra_headers_raw)
    if extra_headers is None:
        st.stop()

    cookies_text = (auth_cookies_raw or "").strip()
    storage_state_path = (auth_storage_state_path or "").strip()
    cookies_list = None
    if cookies_text:
        try:
            parsed = json.loads(cookies_text)
        except json.JSONDecodeError as exc:
            st.error(f"cookies is not valid JSON: {exc}")
            st.stop()
        if not isinstance(parsed, list):
            st.error("cookies must be a JSON array of objects.")
            st.stop()
        cookies_list = parsed

    if cookies_list or storage_state_path:
        auth_body: dict = {"type": "cookie", "extra_headers": extra_headers}
        if cookies_list:
            auth_body["cookies"] = cookies_list
        if storage_state_path:
            auth_body["storage_state_path"] = storage_state_path
        payload["auth"] = auth_body
    elif extra_headers:
        payload["auth"] = {"type": "none", "extra_headers": extra_headers}
    # else: omit auth so backend defaults to AuthNone()

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
        # Sprint 3.2: nudge the user toward the next step.  Saving the
        # target is still a separate click — this is just a pointer,
        # not an automatic save.
        st.info(
            "Looks good. Click **Save** above to persist this target, "
            "then head to **Run test** to launch a full suite against it."
        )
        with st.expander("Response sample (first 200 chars)", expanded=True):
            st.code(sample)
        # Sprint 3.3: show where each secret resolved from (names only,
        # never values). Helps users confirm the vault entry they just
        # pasted is actually the one the adapter consulted.
        sources = result.get("auth_sources") or {}
        if sources:
            with st.expander("Auth sources (where each secret resolved from)", expanded=False):
                st.table(
                    [{"reference": k, "source": v} for k, v in sources.items()]
                )
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

        # Sprint 2.3: when the failure looks like "missing credential"
        # (401/403 over transport, or a config-category error), and we
        # know at least one of the named ``*_env`` references resolves
        # to nothing right now, surface an actionable hint pointing
        # the user at the 🔐 vault field. Re-fetch the vault state here
        # because the user may have just pasted a key in this run; we
        # want the freshest source diagnosis.
        meta_fail = result.get("metadata") or {}
        msg = (result.get("error_message") or "").lower()
        status_code = meta_fail.get("status_code")
        looks_like_auth_miss = (
            cat == "config"
            or status_code in (401, 403)
            or "401" in msg
            or "403" in msg
            or "unauthorized" in msg
            or "forbidden" in msg
        )
        if looks_like_auth_miss:
            # Prefer the backend-supplied auth_sources map (computed
            # from the actual target dict, including any ${VAR} refs
            # inside headers / extra_headers). Fall back to the local
            # vault state walk if the gateway is older.
            backend_sources = result.get("auth_sources") or {}
            if backend_sources:
                missing_refs = [
                    ref for ref, src in backend_sources.items()
                    if src == "missing"
                ]
                if missing_refs:
                    bullets = "\n".join(f"- `{ref}`" for ref in missing_refs)
                    st.warning(
                        "Looks like a credential is missing. Paste the "
                        "value into the 🔐 **Vault value** field next to "
                        "each reference below and re-run **Test connection**:\n\n"
                        + bullets
                    )
            else:
                fresh_vault = _fetch_vault_state()
                named_refs = [
                    ("auth.token_env",         auth_token_env),
                    ("auth.query_value_env",   auth_query_value_env),
                    ("auth.username_env",      auth_basic_user_env),
                    ("auth.password_env",      auth_basic_pass_env),
                    ("auth.headers ${...}",    auth_header_secret_name),
                ]
                missing = [
                    (label, name)
                    for label, name in named_refs
                    if name and fresh_vault.get(name, "missing") == "missing"
                ]
                if missing:
                    bullets = "\n".join(
                        f"- `{label}` → `{name}` has no value yet"
                        for label, name in missing
                    )
                    st.warning(
                        "Looks like a credential is missing.  Paste the secret "
                        "into the 🔐 **Vault value** field next to each "
                        "reference below and re-run **Test connection**:\n\n"
                        + bullets
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
