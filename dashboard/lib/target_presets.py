"""dashboard/lib/target_presets.py
=====================================
Provider-template snippets for the Targets form.

A *preset* is a partial `TargetConfig` dict with the endpoint, auth
shape, request template and response_path pre-filled for a known
provider. The user picks one from a selectbox outside the form, the
page reruns, and the form fields populate from the preset just like
they would from an existing target ("Edit ↓"). The user then names
the secret + pastes the value into the 🔐 vault field — and saves.

This is intentionally **just data** — no provider-specific code paths,
no special-cased adapters. The runtime treats a preset-built target
exactly like any other entry in targets.yaml.

Adding a new preset
-------------------
Drop a new entry into ``TARGET_PRESETS``. Keep secret names generic
(``MY_<PROVIDER>_KEY``) so users see "pick your own name" muscle
memory. Endpoint pinned to the public docs URL at preset-write time;
revisit when the provider rolls a breaking endpoint change.

This module has zero Streamlit / runtime dependencies — pure dict —
so it stays importable from anywhere and is trivial to unit-test.
"""
from __future__ import annotations

from typing import Any, Dict


# ---------------------------------------------------------------------------
# Preset catalog
# ---------------------------------------------------------------------------
# Each preset is a partial-target dict ready for ``TargetConfig.model_validate``
# after a couple of placeholders (id/name + secret value) are filled in.

TARGET_PRESETS: Dict[str, Dict[str, Any]] = {
    # ----- "Pick a starting point" sentinel — not a real preset -----
    "(custom)": {},

    # OpenAI v1 chat completions — Bearer token via ``Authorization``.
    "OpenAI (chat completions)": {
        "type": "api",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "http_method": "POST",
        "auth": {"type": "bearer", "token_env": "MY_OPENAI_KEY"},
        "request_template": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "{prompt}"}],
        },
        "response_path": "choices.0.message.content",
        "timeout_seconds": 60.0,
    },

    # Anthropic Claude — custom header auth (x-api-key + version).
    "Anthropic Claude (messages)": {
        "type": "api",
        "endpoint": "https://api.anthropic.com/v1/messages",
        "http_method": "POST",
        "auth": {
            "type": "header",
            "headers": {
                "x-api-key": "${MY_ANTHROPIC_KEY}",
                "anthropic-version": "2023-06-01",
            },
        },
        "request_template": {
            "model": "claude-sonnet-4-5",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "{prompt}"}],
        },
        "response_path": "content.0.text",
        "timeout_seconds": 60.0,
    },

    # Google Gemini — public docs default to the `?key=` query param.
    "Google Gemini (query key)": {
        "type": "api",
        "endpoint": (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-flash-latest:generateContent"
        ),
        "http_method": "POST",
        "auth": {
            "type": "query",
            "query_key": "key",
            "query_value_env": "MY_GEMINI_KEY",
        },
        "request_template": {
            "contents": [{"parts": [{"text": "{prompt}"}]}],
        },
        "response_path": "candidates.0.content.parts.0.text",
        "timeout_seconds": 30.0,
    },

    # Same provider, header-auth variant — useful for environments that
    # block URL-embedded keys at the proxy.
    "Google Gemini (x-goog-api-key header)": {
        "type": "api",
        "endpoint": (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-flash-latest:generateContent"
        ),
        "http_method": "POST",
        "auth": {
            "type": "header",
            "headers": {"x-goog-api-key": "${MY_GEMINI_KEY}"},
        },
        "request_template": {
            "contents": [{"parts": [{"text": "{prompt}"}]}],
        },
        "response_path": "candidates.0.content.parts.0.text",
        "timeout_seconds": 30.0,
    },

    # Cohere chat — Bearer.
    "Cohere (chat)": {
        "type": "api",
        "endpoint": "https://api.cohere.com/v2/chat",
        "http_method": "POST",
        "auth": {"type": "bearer", "token_env": "MY_COHERE_KEY"},
        "request_template": {
            "model": "command-r-plus",
            "messages": [{"role": "user", "content": "{prompt}"}],
        },
        "response_path": "message.content.0.text",
        "timeout_seconds": 60.0,
    },

    # Mistral chat — Bearer.
    "Mistral (chat completions)": {
        "type": "api",
        "endpoint": "https://api.mistral.ai/v1/chat/completions",
        "http_method": "POST",
        "auth": {"type": "bearer", "token_env": "MY_MISTRAL_KEY"},
        "request_template": {
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "{prompt}"}],
        },
        "response_path": "choices.0.message.content",
        "timeout_seconds": 60.0,
    },

    # Groq (OpenAI-compatible) — Bearer.
    "Groq (OpenAI-compatible)": {
        "type": "api",
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "http_method": "POST",
        "auth": {"type": "bearer", "token_env": "MY_GROQ_KEY"},
        "request_template": {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": "{prompt}"}],
        },
        "response_path": "choices.0.message.content",
        "timeout_seconds": 30.0,
    },

    # ----- Sprint 8: Open WebUI as a self-hosted *web* target -----
    # A ChatGPT-style UI driven by the WebAdapter (Playwright). Backed
    # by the in-network `ollama` service — no paid API, no key.
    #
    # `endpoint` is the IN-NETWORK URL: the WebAdapter runs inside the
    # gateway container, so it reaches the service as `open-webui:8080`
    # (NOT host `localhost:3000`).
    #
    # Selectors verified by live F12 inspection against the TipTap-based
    # Open WebUI build. The response selector pins `nth=-1` so suite runs
    # (which reuse one page across many sends) read the LATEST reply, not
    # the first. Fallbacks cover minor UI revisions (Sprint 5.1).
    #
    # Auth: omitted on purpose — Sprint 8 W1/W2 run with WEBUI_AUTH=False.
    # For the cookie / storage_state path (W4) edit the saved target and
    # switch auth.type to `cookie`.
    "Open WebUI (web)": {
        "type": "web",
        "endpoint": "http://open-webui:8080",
        "timeout_seconds": 60.0,
        "selectors": {
            "input": "#chat-input",
            "submit": "#send-message-button",
            "response": "#response-content-container >> nth=-1",
            "fallback_input": [
                'div[contenteditable="true"]',
            ],
            "fallback_response": [
                "#response-content-container",
                'p[dir="auto"]',
            ],
            "response_wait_ms": 8000,
        },
    },

    # Same Open WebUI instance, REST face. Open WebUI exposes an
    # OpenAI-compatible endpoint at /api/chat/completions guarded by a
    # bearer token. Pairing this with "Open WebUI (web)" gives the
    # thesis a clean A/B: identical backend, two attack surfaces (browser
    # UI vs REST API).
    #
    # Requires WEBUI_AUTH=True + ENABLE_API_KEYS=True in the compose env.
    # Mint the key in Open WebUI → Settings → Account → API Keys, then
    # paste it into the 🔐 vault field on the Targets form.
    #
    # request_template note: Open WebUI's /api/chat/completions wants `stream`
    # (false → single JSON object so `response_path` can extract it). `chat_id`
    # is injected automatically by APIAdapter as a real chat UUID when
    # metadata.openwebui=true, which also mirrors each prompt+reply into a
    # sidebar-visible chat. `model` must match a name from `ollama list`.
    "Open WebUI (api)": {
        "type": "api",
        "endpoint": "http://open-webui:8080/api/chat/completions",
        "http_method": "POST",
        "auth": {"type": "bearer", "token_env": "MY_OPENWEBUI_KEY"},
        "request_template": {
            "model": "qwen2.5:3b",
            "messages": [{"role": "user", "content": "{prompt}"}],
            "stream": False,
        },
        "response_path": "choices.0.message.content",
        "timeout_seconds": 60.0,
        "metadata": {"openwebui": True},
    },
}


def preset_names() -> list:
    """Return the preset labels in catalog order, with the ``(custom)``
    sentinel first so it's the default selection."""
    return list(TARGET_PRESETS.keys())


def get_preset(name: str) -> Dict[str, Any]:
    """Return a *fresh deep copy* of the named preset.

    Returning the same dict from the catalog would let the caller's
    mutations leak into other rerenders. Deep-copy is cheap; presets
    are tiny.
    """
    import copy
    return copy.deepcopy(TARGET_PRESETS.get(name) or {})


__all__ = ["TARGET_PRESETS", "preset_names", "get_preset"]
