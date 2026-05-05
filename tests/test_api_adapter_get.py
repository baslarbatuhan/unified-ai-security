"""tests/test_api_adapter_get.py
==================================
GET-style API adapter contract tests.

Covers the Hafta 9 extension to APIAdapter that supports GET targets
(e.g. the N4B chatbot at `/chat/n4bchatbot?message=...`). Uses httpx's
built-in `MockTransport` so we don't pull a new dependency just for
testing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from external_eval.api_adapter import APIAdapter
from external_eval.base_adapter import AdapterError, AdapterTransportError
from schemas.target_schema import TargetConfig


def _build_target(**overrides) -> TargetConfig:
    base = {
        "id": "hoca_n4b",
        "name": "Hoca N4B Chatbot",
        "type": "api",
        "endpoint": "http://10.147.10.22:9993/chat/n4bchatbot",
        "http_method": "GET",
        "query_template": {"message": "{prompt}"},
        "timeout_seconds": 5.0,
    }
    base.update(overrides)
    return TargetConfig(**base)


def _patch_client(adapter: APIAdapter, handler):
    """Inject a `MockTransport` into the adapter's httpx.Client so each
    request is satisfied by `handler` instead of hitting the network."""
    adapter._client = httpx.Client(
        transport=httpx.MockTransport(handler),
        timeout=adapter.target.timeout_seconds,
    )


# ---------------------------------------------------------------------------
# 1) Plain-text GET success
# ---------------------------------------------------------------------------
def test_get_plain_text_success():
    target = _build_target()
    adapter = APIAdapter(target)

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        return httpx.Response(
            200,
            text="Merhaba! Nasıl yardımcı olabilirim?",
            headers={"content-type": "text/plain; charset=utf-8"},
        )

    _patch_client(adapter, handler)
    resp = adapter.send("Merhaba")

    assert resp.ok, resp.error_message
    assert "Merhaba!" in resp.text
    assert captured["method"] == "GET"
    # httpx URL-encodes the query: ?message=Merhaba (no body sent)
    assert "message=Merhaba" in captured["url"]
    assert resp.metadata["http_method"] == "GET"
    assert resp.metadata["content_type"].startswith("text/plain")


# ---------------------------------------------------------------------------
# 2) GET with non-ASCII prompt — URL-encoding handled by httpx
# ---------------------------------------------------------------------------
def test_get_unicode_prompt_url_encoded():
    target = _build_target()
    adapter = APIAdapter(target)

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, text="ok", headers={"content-type": "text/plain"})

    _patch_client(adapter, handler)
    adapter.send("ığüşöç")

    # httpx percent-encodes the value automatically
    assert "%C4%B1" in captured["url"] or "ığüşöç" in captured["url"]


# ---------------------------------------------------------------------------
# 3) GET with JSON response + response_path extraction
# ---------------------------------------------------------------------------
def test_get_json_response_with_path():
    target = _build_target(response_path="data.reply")
    adapter = APIAdapter(target)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": {"reply": "Hello back"}},
            headers={"content-type": "application/json"},
        )

    _patch_client(adapter, handler)
    resp = adapter.send("Hi")

    assert resp.ok, resp.error_message
    assert resp.text == "Hello back"


# ---------------------------------------------------------------------------
# 4) GET with response_path set on plain-text endpoint → config error
# ---------------------------------------------------------------------------
def test_get_response_path_on_plain_text_is_error():
    target = _build_target(response_path="should.not.be.set")
    adapter = APIAdapter(target)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="raw text", headers={"content-type": "text/plain"})

    _patch_client(adapter, handler)
    resp = adapter.send("Hi")

    # Adapter swallows AdapterError into ChatbotResponse(ok=False).
    assert not resp.ok
    assert resp.error_message is not None
    assert "response_path" in resp.error_message
    assert "not JSON" in resp.error_message


# ---------------------------------------------------------------------------
# 5) Non-2xx GET → AdapterTransportError surfaced as ok=False
# ---------------------------------------------------------------------------
def test_get_non_2xx_surfaces_transport_error():
    target = _build_target()
    adapter = APIAdapter(target)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream unavailable")

    _patch_client(adapter, handler)
    resp = adapter.send("anything")

    assert not resp.ok
    assert "503" in (resp.error_message or "")


# ---------------------------------------------------------------------------
# 6) Schema validator: api+GET requires query_template
# ---------------------------------------------------------------------------
def test_schema_get_without_query_template_rejected():
    with pytest.raises(ValueError, match="query_template"):
        TargetConfig(
            id="bad",
            name="bad",
            type="api",
            endpoint="http://x.y/z",
            http_method="GET",
            # query_template missing
        )


# ---------------------------------------------------------------------------
# 7) Backward compat: existing POST targets still work
# ---------------------------------------------------------------------------
def test_post_still_works_unchanged():
    target = TargetConfig(
        id="legacy_post",
        name="Legacy POST",
        type="api",
        endpoint="http://example.com/v1/chat",
        # http_method default = POST, request_template default applies
        response_path="choices.0.message.content",
    )
    adapter = APIAdapter(target)

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "POST reply"}}]},
            headers={"content-type": "application/json"},
        )

    _patch_client(adapter, handler)
    resp = adapter.send("hi")

    assert resp.ok, resp.error_message
    assert resp.text == "POST reply"
    assert captured["method"] == "POST"
    assert captured["body"]["messages"][0]["content"] == "hi"


# ---------------------------------------------------------------------------
# 8) URL-length pre-flight: oversized prompt → AdapterTransportError
# ---------------------------------------------------------------------------
def test_get_oversized_prompt_pre_flight_error():
    target = _build_target()
    adapter = APIAdapter(target)

    # Don't even need a transport — the check runs before the call.
    huge = "X" * 5000
    resp = adapter.send(huge)

    assert not resp.ok
    assert resp.error_message is not None
    assert "URL would be" in resp.error_message or ">4096" in resp.error_message
