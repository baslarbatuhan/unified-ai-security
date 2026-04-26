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
        auth = self.target.auth or {}
        atype = auth.get("type")
        headers: Dict[str, str] = {}

        if atype == "bearer":
            token = auth.get("token")
            if not token and auth.get("token_env"):
                token = os.environ.get(auth["token_env"], "")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        elif atype == "header":
            extra = auth.get("headers") or {}
            if isinstance(extra, dict):
                for k, v in extra.items():
                    headers[str(k)] = str(v)
        elif atype == "basic":
            # httpx accepts auth tuple, but we emit header so the adapter is
            # uniform and the test path doesn't need httpx.BasicAuth.
            user = auth.get("username", "")
            pw = auth.get("password", "")
            import base64
            token = base64.b64encode(f"{user}:{pw}".encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {token}"
        # unknown / empty → no headers; that's fine for open endpoints.
        return headers

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
        headers = {"Content-Type": "application/json", **self._auth_headers()}
        body = self._render_template(prompt, session_context)
        client = self._get_client()

        try:
            resp = client.post(url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise AdapterTimeout(f"POST {url} timed out after {self.target.timeout_seconds}s") from exc
        except httpx.HTTPError as exc:
            raise AdapterTransportError(f"POST {url} transport error: {exc}") from exc

        if resp.status_code >= 400:
            raise AdapterTransportError(
                f"POST {url} returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise AdapterTransportError(f"POST {url} non-JSON body: {exc}") from exc

        text = self._extract_response(data, self.target.response_path)
        metadata = {
            "status_code": resp.status_code,
            "response_path": self.target.response_path,
            "response_bytes": len(resp.content),
        }
        return text, metadata


__all__ = ["APIAdapter"]
