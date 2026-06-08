"""tests/test_target_schema_cookie.py
=======================================
Sprint 4 — ``AuthCookie`` discriminated-union variant + ``CookieSpec``.

The WebAdapter integration is tested separately (Playwright-gated);
here we cover the pure-Pydantic surface: validation rules, alias
acceptance, ``${VAR}`` value passthrough, and discriminator routing.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.target_schema import (
    AuthCookie,
    CookieSpec,
    TargetConfig,
    TargetsFile,
)


class TestCookieSpec:
    def test_minimal_fields_accepted(self) -> None:
        c = CookieSpec(name="session", value="abc")
        assert c.name == "session"
        assert c.value == "abc"
        # Defaults match Playwright's safe-by-default policy.
        assert c.path == "/"
        assert c.secure is True
        assert c.http_only is False
        assert c.same_site == "Lax"

    def test_playwright_alias_keys_accepted(self) -> None:
        """devtools-exported cookies use ``httpOnly`` / ``sameSite``;
        users can paste them verbatim without renaming."""
        c = CookieSpec(
            name="s", value="v",
            httpOnly=True, sameSite="Strict",
        )
        assert c.http_only is True
        assert c.same_site == "Strict"

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CookieSpec(name="", value="v")

    def test_value_carries_var_placeholder_unchanged(self) -> None:
        """Schema does NOT resolve ``${VAR}`` — it stores the literal so
        the resolver can fan out at request time."""
        c = CookieSpec(name="s", value="${MY_TOKEN}")
        assert c.value == "${MY_TOKEN}"

    def test_invalid_same_site_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CookieSpec(name="s", value="v", same_site="Maybe")  # type: ignore[arg-type]


class TestAuthCookie:
    def test_static_cookies_only(self) -> None:
        a = AuthCookie(type="cookie", cookies=[
            CookieSpec(name="s", value="${MY_S}"),
        ])
        assert len(a.cookies) == 1
        assert a.storage_state_path is None

    def test_storage_state_only(self) -> None:
        a = AuthCookie(type="cookie", storage_state_path="/tmp/state.json")
        assert a.cookies == []
        assert a.storage_state_path == "/tmp/state.json"

    def test_both_sources_combined(self) -> None:
        """User may pin a single cookie on top of a storage_state — used
        to override CSRF or refresh a specific token without re-exporting."""
        a = AuthCookie(
            type="cookie",
            cookies=[CookieSpec(name="csrf", value="${CSRF}")],
            storage_state_path="/tmp/state.json",
        )
        assert len(a.cookies) == 1
        assert a.storage_state_path == "/tmp/state.json"

    def test_no_source_rejected(self) -> None:
        """Empty cookies + no storage_state path is the bug case — both
        misses would silently apply zero auth, masking misconfiguration."""
        with pytest.raises(ValidationError) as exc:
            AuthCookie(type="cookie")
        assert "at least one" in str(exc.value)

    def test_round_trip_through_targetsfile(self) -> None:
        """End-to-end: TargetsFile parses a web target with cookie auth."""
        tf = TargetsFile.model_validate({
            "version": 1,
            "targets": [{
                "id": "web_cookie",
                "name": "Web with cookie",
                "type": "web",
                "endpoint": "https://example.test/chat",
                "selectors": {
                    "input": "textarea",
                    "response": ".msg:last-child",
                },
                "auth": {
                    "type": "cookie",
                    "cookies": [
                        {"name": "session", "value": "${SESSION_TOKEN}"},
                    ],
                },
            }],
        })
        assert len(tf.targets) == 1
        target = tf.targets[0]
        assert target.auth.type == "cookie"
        assert target.auth.cookies[0].name == "session"


class TestDiscriminatorRouting:
    def test_unknown_auth_type_rejected(self) -> None:
        """The discriminated union must not silently accept unknown types
        once cookie is added — guards against typos in user yaml."""
        with pytest.raises(ValidationError):
            TargetConfig(
                id="t", name="T", type="api", endpoint="https://x.test",
                auth={"type": "cookies"},  # extra 's' typo
            )

    def test_cookie_auth_on_api_target_validates(self) -> None:
        """Schema-level permissiveness: cookie auth on an api target
        validates (the dashboard form gates this in UX, but the schema
        stays uniform across variants — mirrors how header/bearer on a
        web target also validate even though the WebAdapter ignores
        them)."""
        t = TargetConfig(
            id="t", name="T", type="api", endpoint="https://x.test",
            auth={"type": "cookie", "cookies": [{"name": "s", "value": "v"}]},
        )
        assert t.auth.type == "cookie"
