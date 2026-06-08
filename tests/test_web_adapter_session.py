"""tests/test_web_adapter_session.py
======================================
Sprint 4 — WebAdapter cookie / storage_state session injection.

We can't launch a real browser in the test suite (Playwright is an
optional dependency that may not even be installed). Two strategies:

1.  ``_resolved_cookies`` and ``_storage_state_path`` are pure helpers —
    we mock the playwright sync_api import just enough to let the
    adapter's ``__init__`` succeed, then call the helpers directly.
2.  Full lifecycle (``_ensure_started``) is exercised against a
    mock Playwright that captures ``new_context`` kwargs and
    ``add_cookies`` calls.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from schemas.target_schema import TargetConfig


# ---------------------------------------------------------------------------
# Fixtures — stub the playwright module before importing WebAdapter so the
# adapter's "playwright is installed" check passes without the real package.
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_playwright(monkeypatch):
    """Install a minimal `playwright.sync_api` stub into sys.modules.

    Returns the mock ``sync_playwright`` callable so tests can introspect
    what the adapter did with the browser/context.
    """
    pw_module = types.ModuleType("playwright")
    sync_api = types.ModuleType("playwright.sync_api")

    sync_playwright = MagicMock(name="sync_playwright")
    sync_api.sync_playwright = sync_playwright  # type: ignore[attr-defined]
    pw_module.sync_api = sync_api  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "playwright", pw_module)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)

    return sync_playwright


@pytest.fixture
def vault(tmp_path: Path, monkeypatch) -> Path:
    p = tmp_path / "secrets.yaml"
    monkeypatch.setenv("UAIS_SECRETS_PATH", str(p))
    return p


def _web_target(**auth) -> TargetConfig:
    return TargetConfig(
        id="w", name="W", type="web",
        endpoint="https://example.test/chat",
        selectors={"input": "textarea", "response": ".msg"},
        auth=auth,
    )


# ---------------------------------------------------------------------------
# _resolved_cookies — value interpolation + url/domain defaulting
# ---------------------------------------------------------------------------
class TestResolvedCookies:
    def test_no_cookie_auth_returns_empty(self, mock_playwright) -> None:
        from external_eval.web_adapter import WebAdapter
        adapter = WebAdapter(_web_target(type="none"))
        assert adapter._resolved_cookies() == []

    def test_literal_value_passes_through(self, mock_playwright) -> None:
        from external_eval.web_adapter import WebAdapter
        adapter = WebAdapter(_web_target(
            type="cookie",
            cookies=[{"name": "s", "value": "literal"}],
        ))
        out = adapter._resolved_cookies()
        assert len(out) == 1
        assert out[0]["name"] == "s"
        assert out[0]["value"] == "literal"
        # Default-bound to the target endpoint URL so users only have
        # to fill name + value for the common case.
        assert out[0]["url"] == "https://example.test/chat"

    def test_var_interpolated_from_env(self, mock_playwright, monkeypatch) -> None:
        from external_eval.web_adapter import WebAdapter
        monkeypatch.setenv("MY_SESSION", "session-from-env")
        adapter = WebAdapter(_web_target(
            type="cookie",
            cookies=[{"name": "session", "value": "${MY_SESSION}"}],
        ))
        out = adapter._resolved_cookies()
        assert out[0]["value"] == "session-from-env"

    def test_var_interpolated_from_vault(self, mock_playwright, vault, monkeypatch) -> None:
        from external_eval import secrets_store
        from external_eval.web_adapter import WebAdapter
        monkeypatch.delenv("VAULT_SESSION", raising=False)
        secrets_store.set("VAULT_SESSION", "session-from-vault")
        adapter = WebAdapter(_web_target(
            type="cookie",
            cookies=[{"name": "session", "value": "${VAULT_SESSION}"}],
        ))
        out = adapter._resolved_cookies()
        assert out[0]["value"] == "session-from-vault"

    def test_missing_var_becomes_empty_string(self, mock_playwright, monkeypatch) -> None:
        """Unresolved ``${VAR}`` → empty value, same contract as bearer/
        query/basic.  Cookie still emitted (downstream auth check fails
        visibly instead of silent skip)."""
        from external_eval.web_adapter import WebAdapter
        monkeypatch.delenv("NOT_SET", raising=False)
        adapter = WebAdapter(_web_target(
            type="cookie",
            cookies=[{"name": "s", "value": "${NOT_SET}"}],
        ))
        out = adapter._resolved_cookies()
        assert out[0]["value"] == ""

    def test_explicit_domain_preserved(self, mock_playwright) -> None:
        """User-provided `domain` survives endpoint defaulting."""
        from external_eval.web_adapter import WebAdapter
        adapter = WebAdapter(_web_target(
            type="cookie",
            cookies=[{
                "name": "s", "value": "v",
                "domain": ".example.test",
            }],
        ))
        out = adapter._resolved_cookies()
        assert out[0].get("domain") == ".example.test"
        assert "url" not in out[0]

    def test_playwright_alias_keys_propagated(self, mock_playwright) -> None:
        """sameSite / httpOnly round-trip from devtools-style JSON to
        Playwright add_cookies payload."""
        from external_eval.web_adapter import WebAdapter
        adapter = WebAdapter(_web_target(
            type="cookie",
            cookies=[{
                "name": "s", "value": "v",
                "httpOnly": True, "sameSite": "Strict",
            }],
        ))
        out = adapter._resolved_cookies()
        assert out[0]["httpOnly"] is True
        assert out[0]["sameSite"] == "Strict"


# ---------------------------------------------------------------------------
# _storage_state_path — passthrough only when configured
# ---------------------------------------------------------------------------
class TestStorageStatePath:
    def test_no_cookie_auth_returns_none(self, mock_playwright) -> None:
        from external_eval.web_adapter import WebAdapter
        adapter = WebAdapter(_web_target(type="none"))
        assert adapter._storage_state_path() is None

    def test_path_propagated(self, mock_playwright) -> None:
        from external_eval.web_adapter import WebAdapter
        adapter = WebAdapter(_web_target(
            type="cookie",
            storage_state_path="/tmp/state.json",
        ))
        assert adapter._storage_state_path() == "/tmp/state.json"


# ---------------------------------------------------------------------------
# Full lifecycle — verify _ensure_started wires cookies + storage_state
# into the Playwright context the way the adapter promises.
# ---------------------------------------------------------------------------
class TestEnsureStartedWiring:
    def _build_pw_chain(self, mock_playwright):
        """Return (page, context, browser) mocks pre-wired so
        ``sync_playwright().start().chromium.launch().new_context().new_page()``
        resolves to ``page`` and the call chain is observable."""
        page = MagicMock(name="page")
        context = MagicMock(name="context")
        context.new_page.return_value = page
        browser = MagicMock(name="browser")
        browser.new_context.return_value = context
        chromium = MagicMock(name="chromium")
        chromium.launch.return_value = browser
        pw_handle = MagicMock(name="pw_handle")
        pw_handle.chromium = chromium
        mock_playwright.return_value.start.return_value = pw_handle
        return page, context, browser

    def test_cookies_passed_to_add_cookies(self, mock_playwright, monkeypatch) -> None:
        from external_eval.web_adapter import WebAdapter
        monkeypatch.setenv("S", "secret")
        page, context, browser = self._build_pw_chain(mock_playwright)

        adapter = WebAdapter(_web_target(
            type="cookie",
            cookies=[{"name": "session", "value": "${S}"}],
        ))
        adapter._ensure_started()

        # storage_state was NOT requested → new_context called with no kwargs.
        assert browser.new_context.call_args.kwargs == {}
        # add_cookies received the resolved value (vault/env, not the placeholder).
        cookies_arg = context.add_cookies.call_args.args[0]
        assert cookies_arg[0]["value"] == "secret"
        # Page.goto fired on the endpoint URL.
        page.goto.assert_called_once()

    def test_storage_state_passed_to_new_context(self, mock_playwright) -> None:
        from external_eval.web_adapter import WebAdapter
        page, context, browser = self._build_pw_chain(mock_playwright)

        adapter = WebAdapter(_web_target(
            type="cookie",
            storage_state_path="/tmp/exported.json",
        ))
        adapter._ensure_started()

        assert browser.new_context.call_args.kwargs == {
            "storage_state": "/tmp/exported.json",
        }
        # No cookie list → add_cookies should NOT have been called.
        context.add_cookies.assert_not_called()

    def test_storage_state_plus_cookies_layered(self, mock_playwright, monkeypatch) -> None:
        """When both are configured, storage_state loads first and the
        explicit cookies override on top — useful for a CSRF refresh
        without re-exporting the whole state."""
        from external_eval.web_adapter import WebAdapter
        monkeypatch.setenv("CSRF", "fresh-csrf")
        page, context, browser = self._build_pw_chain(mock_playwright)

        adapter = WebAdapter(_web_target(
            type="cookie",
            storage_state_path="/tmp/base.json",
            cookies=[{"name": "csrf", "value": "${CSRF}"}],
        ))
        adapter._ensure_started()

        assert browser.new_context.call_args.kwargs["storage_state"] == "/tmp/base.json"
        context.add_cookies.assert_called_once()
        assert context.add_cookies.call_args.args[0][0]["value"] == "fresh-csrf"

    def test_non_cookie_auth_skips_session_injection(self, mock_playwright) -> None:
        from external_eval.web_adapter import WebAdapter
        page, context, browser = self._build_pw_chain(mock_playwright)

        adapter = WebAdapter(_web_target(type="none"))
        adapter._ensure_started()

        assert browser.new_context.call_args.kwargs == {}
        context.add_cookies.assert_not_called()


# ---------------------------------------------------------------------------
# Defensive — clear error message when Playwright is missing entirely.
# ---------------------------------------------------------------------------
def test_missing_playwright_raises_actionable_error(monkeypatch) -> None:
    """Without a Playwright install, WebAdapter construction must fail
    with a message that includes the install command — not a generic
    ImportError."""
    # Ensure the real (or any cached) playwright module is hidden so
    # the adapter's ``try: import playwright.sync_api`` actually raises.
    for k in list(sys.modules):
        if k.startswith("playwright"):
            monkeypatch.delitem(sys.modules, k, raising=False)
    monkeypatch.setattr(
        "builtins.__import__",
        _block_playwright_import(__builtins__["__import__"]
                                 if isinstance(__builtins__, dict)
                                 else __import__),
    )

    from external_eval.web_adapter import WebAdapter
    from external_eval.base_adapter import AdapterConfigError

    with pytest.raises(AdapterConfigError) as exc:
        WebAdapter(TargetConfig(
            id="w", name="W", type="web",
            endpoint="https://example.test",
            selectors={"input": "t", "response": "r"},
        ))
    assert "Playwright" in str(exc.value)
    assert "pip install" in str(exc.value)


def _block_playwright_import(real_import):
    def _imp(name, *args, **kwargs):
        if name.startswith("playwright"):
            raise ImportError("playwright not installed (test stub)")
        return real_import(name, *args, **kwargs)
    return _imp
