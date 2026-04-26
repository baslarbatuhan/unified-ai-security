"""dashboard/lib/gateway_client.py
Thin HTTP client that wraps the gateway's read-only routes.

Centralised here so each Streamlit page calls ``client.get_json("/runs")``
instead of repeating ``requests.get`` boilerplate. Errors propagate as
``GatewayError`` so pages can render a friendly message without seeing
``requests`` details.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
import streamlit as st


_DEFAULT_BASE = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8000").rstrip("/")
# LLM judge calls (rag_guard, prompt_guard) can take 30–60 s under load.
# 120 s keeps the dashboard responsive without killing real analysis calls.
# Override with GATEWAY_TIMEOUT_S env var for strict environments.
_DEFAULT_TIMEOUT = float(os.environ.get("GATEWAY_TIMEOUT_S", "120"))


class GatewayError(RuntimeError):
    """Raised when the gateway returns a non-2xx response or is unreachable."""


@dataclass
class GatewayClient:
    base_url: str = _DEFAULT_BASE
    timeout_s: float = _DEFAULT_TIMEOUT

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    def get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        try:
            resp = requests.get(self._url(path), params=params, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise GatewayError(f"GET {path} failed: {exc}") from exc
        if resp.status_code >= 400:
            raise GatewayError(f"GET {path} → HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json() if resp.content else None

    def get_text(self, path: str, params: Optional[Dict[str, Any]] = None) -> str:
        try:
            resp = requests.get(self._url(path), params=params, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise GatewayError(f"GET {path} failed: {exc}") from exc
        if resp.status_code >= 400:
            raise GatewayError(f"GET {path} → HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.text

    def get_bytes(self, path: str, params: Optional[Dict[str, Any]] = None) -> bytes:
        try:
            resp = requests.get(self._url(path), params=params, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise GatewayError(f"GET {path} failed: {exc}") from exc
        if resp.status_code >= 400:
            raise GatewayError(f"GET {path} → HTTP {resp.status_code}")
        return resp.content

    def post_json(self, path: str, payload: Dict[str, Any]) -> Any:
        try:
            resp = requests.post(self._url(path), json=payload, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise GatewayError(f"POST {path} failed: {exc}") from exc
        if resp.status_code >= 400:
            raise GatewayError(f"POST {path} → HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json() if resp.content else None

    def delete(self, path: str) -> Any:
        try:
            resp = requests.delete(self._url(path), timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise GatewayError(f"DELETE {path} failed: {exc}") from exc
        if resp.status_code >= 400:
            raise GatewayError(f"DELETE {path} → HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json() if resp.content else None


@st.cache_resource(show_spinner=False)
def get_default_client() -> GatewayClient:
    """Return a process-wide ``GatewayClient`` so pages share connection state."""
    return GatewayClient()
