"""external_eval/api_adapter.py
=================================
REST adapter.  Sends a prompt to a JSON HTTP endpoint and extracts the
assistant reply via a dot-path response template.

Design
------
* Uses `httpx` (sync).  httpx is already a transitive dep via FastAPI /
  test client; we don't pull new packages.
* Auth:
    - `{type: bearer, token: "..."}`     — Authorization: Bearer
    - `{type: bearer, token_env: "..."}` — token read from env var
    - `{type: header, headers: {...}}`   — arbitrary headers
    - `{type: basic, username, password}` — HTTP basic
* Request body: rendered from `target.request_template`; the substring
  `{prompt}` (and, if present, `{role}`) is replaced.  Template is a
  Python dict — the replacement walks strings.
* Response extraction: `target.response_path` is a dot-path like
  `choices.0.message.content`.  List indices are integers.

Errors:
  * non-2xx       → `AdapterTransportError`
  * timeout       → `AdapterTimeout`
  * path missing  → `AdapterError`
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, Optional, Tuple

import httpx

from schemas.target_schema import TargetConfig
from external_eval.base_adapter import (
    AdapterConfigError,
    AdapterError,
    AdapterTimeout,
    AdapterTransportError,
    ChatbotAdapter,
)


DEFAULT_REQUEST_TEMPLATE: Dict[str, Any] = {
    "messages": [{"role": "user", "content": "{prompt}"}]
}


class APIAdapter(ChatbotAdapter):
    def __init__(self, target: TargetConfig):
        super().__init__(target)
        if target.type != "api":
            raise AdapterConfigError(
                f"APIAdapter requires type='api', got {target.type!r}"
            )
        if not target.endpoint:
            raise AdapterConfigError(
                f"APIAdapter target {target.id!r} missing endpoint"
            )
        self._client: Optional[httpx.Client] = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.target.timeout_seconds)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _auth_headers(self) -> Dict[str, str]:
        """Build the per-request header set from `target.auth`.

        Hafta 11.2: dispatch on the discriminated union's `.type` literal.
        `extra_headers` from any auth variant (including `none`) is merged
        last so vendor-specific static headers (OpenAI org id, tenant id,
        Accept-Version pinning) always make it onto the request.

        Returns an empty dict for `none` auth + empty extras — that's fine
        for open endpoints; the caller adds Content-Type as needed.
        """
        auth = self.target.auth
        atype = getattr(auth, "type", None)
        headers: Dict[str, str] = {}

        if atype == "bearer":
            # Prefer env-var lookup; fall back to inline `token` for tests
            # and the legacy YAML shape.
            token = (auth.token or "")
            if not token and auth.token_env:
                token = os.environ.get(auth.token_env, "")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        elif atype == "header":
            for k, v in (auth.headers or {}).items():
                headers[str(k)] = str(v)

        elif atype == "basic":
            user = auth.username or (os.environ.get(auth.username_env, "") if auth.username_env else "")
            pw = auth.password or (os.environ.get(auth.password_env, "") if auth.password_env else "")
            if user or pw:
                import base64
                token = base64.b64encode(f"{user}:{pw}".encode("utf-8")).decode("ascii")
                headers["Authorization"] = f"Basic {token}"

        # `query` auth doesn't add headers — it's applied to the URL via
        # `_apply_query_auth` at request time.

        # Always merge extra_headers (works for any auth type, including
        # `none`). Caller-set headers take precedence over extras, so we
        # write extras first.
        merged = dict(getattr(auth, "extra_headers", {}) or {})
        merged.update(headers)
        return {str(k): str(v) for k, v in merged.items()}

    def _apply_query_auth(self, params: Dict[str, str]) -> Dict[str, str]:
        """For `auth.type=query`, merge the auth key/value into the query
        params dict. No-op for other auth types. The query param is added
        even if the caller's `_render_query()` didn't produce one — handy
        for endpoints that take only the auth key (no prompt body).
        """
        auth = self.target.auth
        if getattr(auth, "type", None) != "query":
            return params
        value = auth.query_value or ""
        if not value and auth.query_value_env:
            value = os.environ.get(auth.query_value_env, "")
        if not value:
            # Missing secret at runtime — surface as empty param so the
            # downstream 401/403 is the visible failure mode instead of a
            # silent skip. Logs will show the missing env var.
            value = ""
        # Caller-provided params win on collision (extremely unlikely; we
        # don't want auth to overwrite a `key` param the template emitted).
        out = {auth.query_key: value}
        out.update(params)
        return out

    def _render_template(self, prompt: str, session_context: Dict[str, Any]) -> Dict[str, Any]:
        tpl = copy.deepcopy(self.target.request_template or DEFAULT_REQUEST_TEMPLATE)
        role = str(session_context.get("role", "user"))

        def _walk(node: Any) -> Any:
            if isinstance(node, str):
                return node.replace("{prompt}", prompt).replace("{role}", role)
            if isinstance(node, dict):
                return {k: _walk(v) for k, v in node.items()}
            if isinstance(node, list):
                return [_walk(v) for v in node]
            return node

        return _walk(tpl)

    def _render_query(self, prompt: str, session_context: Dict[str, Any]) -> Dict[str, str]:
        """Render `query_template` for GET-style targets.

        httpx.Client.get(params=...) accepts a flat {str: str} dict and
        URL-encodes values automatically. We only walk one level deep —
        nested dicts are intentionally rejected because URL query
        strings are flat. Use POST + request_template if you need a
        nested payload.
        """
        tpl = self.target.query_template or {}
        if not isinstance(tpl, dict):
            raise AdapterConfigError(
                f"target {self.target.id!r}: query_template must be a flat dict"
            )
        role = str(session_context.get("role", "user"))
        rendered: Dict[str, str] = {}
        for k, v in tpl.items():
            sv = str(v)
            sv = sv.replace("{prompt}", prompt).replace("{role}", role)
            rendered[str(k)] = sv
        return rendered

    @staticmethod
    def _extract_response(data: Any, path: Optional[str]) -> str:
        if not path:
            # No explicit path: try a few conventional shapes before giving up.
            for candidate in (
                ("choices", 0, "message", "content"),
                ("message", "content"),
                ("content",),
                ("output",),
                ("text",),
            ):
                cur: Any = data
                ok = True
                for seg in candidate:
                    try:
                        cur = cur[seg] if isinstance(seg, int) else cur[seg]
                    except (KeyError, IndexError, TypeError):
                        ok = False
                        break
                if ok and isinstance(cur, str):
                    return cur
            # Fall through → stringify.
            return json.dumps(data, ensure_ascii=False)

        cur: Any = data
        for seg in path.split("."):
            if seg.isdigit() and isinstance(cur, list):
                idx = int(seg)
                if idx >= len(cur):
                    raise AdapterError(f"response_path {path!r}: index {idx} out of range")
                cur = cur[idx]
            elif isinstance(cur, dict):
                if seg not in cur:
                    raise AdapterError(f"response_path {path!r}: key {seg!r} missing")
                cur = cur[seg]
            else:
                raise AdapterError(f"response_path {path!r}: cannot descend into {type(cur).__name__}")
        if not isinstance(cur, str):
            cur = json.dumps(cur, ensure_ascii=False)
        return cur

    # ------------------------------------------------------------------
    # ChatbotAdapter API
    # ------------------------------------------------------------------
    def _send_impl(
        self, prompt: str, session_context: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        url = self.target.endpoint or ""
        method = (getattr(self.target, "http_method", "POST") or "POST").upper()
        headers = self._auth_headers()
        client = self._get_client()

        try:
            if method == "GET":
                params = self._render_query(prompt, session_context)
                # Hafta 11.2: append the auth.query key/value when
                # auth.type=query. No-op for other auth types so existing
                # GET targets are unaffected.
                params = self._apply_query_auth(params)
                # Pre-flight URL length sanity check — query strings get
                # truncated by gateways/CDNs around 2-4 KB. Surface this
                # as a config error (not a transport error) so the runner
                # can attribute the failure to the prompt size, not the
                # network.
                approx_len = len(url) + sum(len(k) + len(v) + 2 for k, v in params.items())
                if approx_len > 4096:
                    raise AdapterTransportError(
                        f"GET {url}: request URL would be ~{approx_len} bytes "
                        "(>4096); switch to POST + request_template."
                    )
                resp = client.get(url, headers=headers, params=params)
            else:  # POST default — backward compatible
                headers["Content-Type"] = "application/json"
                body = self._render_template(prompt, session_context)
                # Hafta 11.2: even POST endpoints may use query auth
                # (e.g. Gemini's generateContent uses ?key=… alongside a
                # JSON body). For auth types other than `query` this is
                # an empty dict and httpx omits the query string.
                params = self._apply_query_auth({})
                resp = client.post(url, headers=headers, json=body, params=params or None)
        except httpx.TimeoutException as exc:
            raise AdapterTimeout(
                f"{method} {url} timed out after {self.target.timeout_seconds}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise AdapterTransportError(f"{method} {url} transport error: {exc}") from exc

        if resp.status_code >= 400:
            raise AdapterTransportError(
                f"{method} {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

        # Content-type aware response handling — chatbots that respond
        # in plain text (e.g. simple GET endpoints) shouldn't be forced
        # through `resp.json()`. JSON path still tried first when
        # Content-Type advertises JSON.
        ctype = (resp.headers.get("content-type") or "").lower()
        is_json = "application/json" in ctype or ctype.startswith("application/")
        if is_json:
            try:
                data = resp.json()
            except ValueError as exc:
                raise AdapterTransportError(
                    f"{method} {url} advertised JSON but body unparseable: {exc}"
                ) from exc
            text = self._extract_response(data, self.target.response_path)
        else:
            # Plain text / unknown content-type — return body as-is. If
            # the operator set response_path on a non-JSON endpoint, that
            # is a config mistake; surface it loudly rather than silently
            # ignoring.
            if self.target.response_path:
                raise AdapterError(
                    f"{method} {url}: response_path={self.target.response_path!r} "
                    f"set but Content-Type is {ctype!r} (not JSON). "
                    "Clear response_path for plain-text endpoints."
                )
            text = resp.text

        metadata = {
            "status_code": resp.status_code,
            "http_method": method,
            "response_path": self.target.response_path,
            "response_bytes": len(resp.content),
            "content_type": ctype,
        }
        return text, metadata


__all__ = ["APIAdapter"]
