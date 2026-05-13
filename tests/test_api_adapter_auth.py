"""tests/test_api_adapter_auth.py
==================================
Hafta 11.2 — APIAdapter auth header / query injection across 5 variants.

We don't actually hit a network — `_auth_headers()` and
`_apply_query_auth()` are pure functions of the TargetConfig. For the
end-to-end `send()` path we patch httpx.Client at the module level so we
can capture the final request without a real server.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from external_eval.api_adapter import APIAdapter
from schemas.target_schema import TargetConfig


def _api_target(**auth_kwargs) -> TargetConfig:
    """Build a minimal API target with the given auth shape."""
    return TargetConfig(
        id="t", name="T", type="api", endpoint="https://example.test/chat",
        auth=auth_kwargs,
    )


# ---------------------------------------------------------------------------
# _auth_headers — per variant
# ---------------------------------------------------------------------------
class TestAuthHeaders:
    def test_none_yields_empty(self) -> None:
        adapter = APIAdapter(_api_target(type="none"))
        assert adapter._auth_headers() == {}

    def test_bearer_inline_token(self) -> None:
        adapter = APIAdapter(_api_target(type="bearer", token="sk-test"))
        assert adapter._auth_headers() == {"Authorization": "Bearer sk-test"}

    def test_bearer_env_lookup(self, monkeypatch) -> None:
        monkeypatch.setenv("UAIS_TEST_BEARER", "env-secret")
        adapter = APIAdapter(_api_target(type="bearer", token_env="UAIS_TEST_BEARER"))
        assert adapter._auth_headers()["Authorization"] == "Bearer env-secret"

    def test_bearer_missing_env_emits_no_header(self, monkeypatch) -> None:
        """Missing env var → no Authorization (downstream 401 surfaces it)."""
        monkeypatch.delenv("UAIS_TEST_BEARER_ABSENT", raising=False)
        adapter = APIAdapter(_api_target(
            type="bearer", token_env="UAIS_TEST_BEARER_ABSENT",
        ))
        h = adapter._auth_headers()
        assert "Authorization" not in h

    def test_header_variant_copies_headers(self) -> None:
        adapter = APIAdapter(_api_target(
            type="header",
            headers={"X-API-Key": "lit-key", "X-Tenant": "t1"},
        ))
        h = adapter._auth_headers()
        assert h["X-API-Key"] == "lit-key"
        assert h["X-Tenant"] == "t1"

    def test_basic_emits_base64_authorization(self) -> None:
        adapter = APIAdapter(_api_target(
            type="basic", username="alice", password="pw",
        ))
        import base64
        expected = base64.b64encode(b"alice:pw").decode("ascii")
        assert adapter._auth_headers()["Authorization"] == f"Basic {expected}"

    def test_basic_with_env_lookup(self, monkeypatch) -> None:
        monkeypatch.setenv("UAIS_TEST_BASIC_USER", "u_env")
        monkeypatch.setenv("UAIS_TEST_BASIC_PASS", "p_env")
        adapter = APIAdapter(_api_target(
            type="basic",
            username_env="UAIS_TEST_BASIC_USER",
            password_env="UAIS_TEST_BASIC_PASS",
        ))
        import base64
        expected = base64.b64encode(b"u_env:p_env").decode("ascii")
        assert adapter._auth_headers()["Authorization"] == f"Basic {expected}"

    def test_query_variant_adds_no_header(self) -> None:
        """query auth lives in the URL, never as a header."""
        adapter = APIAdapter(_api_target(
            type="query", query_key="key", query_value="literal",
        ))
        assert adapter._auth_headers() == {}

    # ---- extra_headers behaviour across all types ----
    def test_extra_headers_merged_with_none(self) -> None:
        adapter = APIAdapter(_api_target(
            type="none", extra_headers={"X-Tenant": "t1"},
        ))
        assert adapter._auth_headers()["X-Tenant"] == "t1"

    def test_extra_headers_merged_with_bearer(self) -> None:
        adapter = APIAdapter(_api_target(
            type="bearer", token="sk-x",
            extra_headers={"OpenAI-Organization": "org-xyz"},
        ))
        h = adapter._auth_headers()
        assert h["Authorization"] == "Bearer sk-x"
        assert h["OpenAI-Organization"] == "org-xyz"

    def test_explicit_auth_header_overrides_extra(self) -> None:
        """A bearer Authorization header must not be overwritten by an
        extra_headers entry of the same name (precedence: variant wins)."""
        adapter = APIAdapter(_api_target(
            type="bearer", token="sk-x",
            extra_headers={"Authorization": "Bearer SHOULD-NOT-WIN"},
        ))
        assert adapter._auth_headers()["Authorization"] == "Bearer sk-x"


# ---------------------------------------------------------------------------
# _apply_query_auth — URL param injection
# ---------------------------------------------------------------------------
class TestApplyQueryAuth:
    def test_noop_for_non_query_types(self) -> None:
        adapter = APIAdapter(_api_target(type="bearer", token="x"))
        assert adapter._apply_query_auth({"message": "hi"}) == {"message": "hi"}

    def test_injects_query_key_value(self, monkeypatch) -> None:
        monkeypatch.setenv("UAIS_GEMINI_KEY", "gem-secret")
        adapter = APIAdapter(_api_target(
            type="query", query_key="key", query_value_env="UAIS_GEMINI_KEY",
        ))
        out = adapter._apply_query_auth({"message": "hi"})
        assert out == {"key": "gem-secret", "message": "hi"}

    def test_caller_params_win_on_collision(self) -> None:
        """If the template already provides `key`, the auth value must NOT
        clobber it (extremely defensive — we don't want a query param
        named `key` from the template to get overridden silently)."""
        adapter = APIAdapter(_api_target(
            type="query", query_key="key", query_value="auth-val",
        ))
        out = adapter._apply_query_auth({"key": "template-val"})
        assert out["key"] == "template-val"

    def test_missing_env_yields_empty_value(self, monkeypatch) -> None:
        monkeypatch.delenv("UAIS_MISSING_QV", raising=False)
        adapter = APIAdapter(_api_target(
            type="query", query_key="key", query_value_env="UAIS_MISSING_QV",
        ))
        # Param key is still present so the 401/403 fires server-side,
        # which is the visible failure mode we want (vs silent skip).
        out = adapter._apply_query_auth({"message": "hi"})
        assert "key" in out
        assert out["key"] == ""


# ---------------------------------------------------------------------------
# End-to-end send() — captured httpx call carries the right headers + params
# ---------------------------------------------------------------------------
class TestSendIntegration:
    def _capture_client(self, status_code: int = 200, body_json: Any = None):
        """Make a stand-in httpx.Client whose .post / .get capture the
        keyword args the adapter passed. Returns (mock_client, captured)."""
        captured: Dict[str, Any] = {}

        def _make_resp():
            r = MagicMock()
            r.status_code = status_code
            r.json.return_value = body_json or {"response": "ok"}
            r.text = "ok"
            return r

        def _post(url, **kwargs):
            captured.update({"method": "POST", "url": url, **kwargs})
            return _make_resp()

        def _get(url, **kwargs):
            captured.update({"method": "GET", "url": url, **kwargs})
            return _make_resp()

        client = MagicMock()
        client.post = _post
        client.get = _get
        return client, captured

    def test_query_auth_injects_param_on_get(self, monkeypatch) -> None:
        monkeypatch.setenv("UAIS_TEST_KEY", "k-secret")
        target = TargetConfig(
            id="g", name="G", type="api", endpoint="https://x.test/v1",
            http_method="GET",
            query_template={"q": "{prompt}"},
            response_path="text",
            auth={"type": "query", "query_key": "key",
                  "query_value_env": "UAIS_TEST_KEY"},
        )
        adapter = APIAdapter(target)
        client, captured = self._capture_client(body_json={"text": "hi"})
        adapter._client = client  # bypass _get_client() construction

        adapter.send("hello")
        assert captured["method"] == "GET"
        # Auth param appears alongside template-rendered params.
        params = captured["params"]
        assert params["key"] == "k-secret"
        assert params["q"] == "hello"

    def test_extra_headers_attached_on_post(self) -> None:
        target = TargetConfig(
            id="o", name="O", type="api", endpoint="https://o.test/v1/chat",
            http_method="POST",
            auth={
                "type": "bearer", "token": "sk-test",
                "extra_headers": {"OpenAI-Organization": "org-z"},
            },
        )
        adapter = APIAdapter(target)
        client, captured = self._capture_client(body_json={"response": "hi"})
        adapter._client = client

        adapter.send("hello")
        assert captured["method"] == "POST"
        headers = captured["headers"]
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["OpenAI-Organization"] == "org-z"
        assert headers["Content-Type"] == "application/json"

    def test_query_auth_on_post_endpoint(self, monkeypatch) -> None:
        """Gemini-style: POST body + ?key=… auth in the URL."""
        monkeypatch.setenv("UAIS_GEMINI_KEY", "gem")
        target = TargetConfig(
            id="gem", name="G", type="api",
            endpoint="https://generativelanguage.googleapis.test/v1/x",
            http_method="POST",
            request_template={"contents": [{"parts": [{"text": "{prompt}"}]}]},
            auth={"type": "query", "query_key": "key",
                  "query_value_env": "UAIS_GEMINI_KEY"},
        )
        adapter = APIAdapter(target)
        client, captured = self._capture_client(body_json={"response": "hi"})
        adapter._client = client

        adapter.send("hello")
        assert captured["method"] == "POST"
        assert captured.get("params") == {"key": "gem"}
