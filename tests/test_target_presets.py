"""tests/test_target_presets.py
=================================
Sprint 6 — every entry in ``dashboard/lib/target_presets.TARGET_PRESETS``
must validate against the live ``TargetConfig`` schema after the user
supplies the two missing fields (``id`` + final secret value).

We intentionally do NOT pin the request templates / endpoints in
assertions — those are policy that changes as providers rev their
APIs. The contract is "schema-valid + non-empty critical fields".
"""
from __future__ import annotations

import pytest

from dashboard.lib.target_presets import (
    TARGET_PRESETS,
    get_preset,
    preset_names,
)
from schemas.target_schema import TargetConfig


def _names_excluding_sentinel():
    """Names of real presets — drops the ``(custom)`` placeholder."""
    return [n for n in preset_names() if n != "(custom)"]


class TestCatalog:
    def test_custom_sentinel_present_and_first(self) -> None:
        assert preset_names()[0] == "(custom)"
        assert get_preset("(custom)") == {}

    def test_at_least_a_handful_of_presets(self) -> None:
        """Sanity floor — keep the catalog meaningful. Update as the
        list grows or shrinks intentionally."""
        assert len(_names_excluding_sentinel()) >= 5

    def test_get_preset_returns_deep_copy(self) -> None:
        """Mutating one caller's copy must not pollute the catalog."""
        a = get_preset("OpenAI (chat completions)")
        a["endpoint"] = "https://evil.test"
        b = get_preset("OpenAI (chat completions)")
        assert b["endpoint"] != "https://evil.test"


class TestSchemaCompliance:
    @pytest.mark.parametrize("name", _names_excluding_sentinel())
    def test_preset_validates_with_dummy_id(self, name: str) -> None:
        """Filling in id+name reaches a TargetConfig that round-trips."""
        preset = get_preset(name)
        preset["id"] = "fixture_t"
        preset.setdefault("name", name)
        # Should NOT raise — every preset is a complete target shape
        # apart from the user's own naming.
        target = TargetConfig.model_validate(preset)
        assert target.id == "fixture_t"
        # Pre-screen sanity: api presets must declare endpoint + method.
        if target.type == "api":
            assert target.endpoint
            assert target.http_method in ("POST", "GET")

    @pytest.mark.parametrize("name", _names_excluding_sentinel())
    def test_preset_has_response_path_for_api(self, name: str) -> None:
        """API presets always include a response_path so the adapter
        knows how to extract the assistant text. Web/cookie presets
        wouldn't need it but none exist yet."""
        preset = get_preset(name)
        if preset.get("type") != "api":
            pytest.skip("non-api preset")
        assert preset.get("response_path"), (
            f"preset {name!r} is api-typed but missing response_path; "
            "adapters will default to plain-text mode."
        )

    @pytest.mark.parametrize("name", _names_excluding_sentinel())
    def test_no_inline_secrets_in_preset(self, name: str) -> None:
        """Defence-in-depth: a preset must not ship with literal token
        values. Either ``*_env`` references or ``${VAR}`` placeholders;
        never a real secret string. Cheap to catch in CI."""
        preset = get_preset(name)
        auth = preset.get("auth") or {}
        atype = auth.get("type")
        if atype == "bearer":
            assert not auth.get("token"), f"{name}: inline `token` would leak"
        if atype == "query":
            assert not auth.get("query_value"), f"{name}: inline `query_value` would leak"
        if atype == "basic":
            assert not auth.get("password"), f"{name}: inline `password` would leak"
        if atype == "header":
            # Header values may legitimately carry literals (e.g.,
            # anthropic-version: 2023-06-01).  We just disallow values
            # that look like real API keys (typical OpenAI sk-/ant-/
            # AIzaSy- prefixes).
            for hv in (auth.get("headers") or {}).values():
                low = str(hv).lower()
                assert not low.startswith(("sk-", "ant-", "aizasy")), (
                    f"{name}: header value looks like a real API key"
                )


class TestSecretNameConvention:
    @pytest.mark.parametrize("name", _names_excluding_sentinel())
    def test_secret_names_follow_my_convention(self, name: str) -> None:
        """Presets reference env-var names like ``MY_<PROVIDER>_KEY`` to
        steer users toward "you pick the name" — not the legacy
        provider-canonical names (OPENAI_API_KEY etc.) we worked to
        decouple in Faz 1."""
        preset = get_preset(name)
        auth = preset.get("auth") or {}
        atype = auth.get("type")
        env_refs = []
        if atype == "bearer" and auth.get("token_env"):
            env_refs.append(auth["token_env"])
        if atype == "query" and auth.get("query_value_env"):
            env_refs.append(auth["query_value_env"])
        if atype == "basic":
            if auth.get("username_env"):
                env_refs.append(auth["username_env"])
            if auth.get("password_env"):
                env_refs.append(auth["password_env"])
        if atype == "header":
            # Pull ${VAR} names from header values.
            import re
            for hv in (auth.get("headers") or {}).values():
                env_refs.extend(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", str(hv)))
        # Each env-var name a preset references should start with MY_
        # so the user sees the rename-me hint.
        for ref in env_refs:
            assert ref.startswith("MY_"), (
                f"{name}: env-var ref {ref!r} does not start with MY_; "
                "discourages the 'pick your own name' UX."
            )
