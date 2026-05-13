"""tests/test_target_schema_auth.py
====================================
Hafta 11.2 — AuthConfig discriminated union + redaction.

Covers:
  * 5 auth variants (none / bearer / header / query / basic) construct
    via Pydantic with the right `type` literal.
  * Variant-specific validators reject incomplete shapes.
  * `TargetConfig._normalize_auth` swallows the legacy {} / missing-type
    cases so existing targets.yaml files keep parsing.
  * `_redact_auth` strips `token`, `password`, `query_value`, and
    credential-looking headers — defence-in-depth on every read path.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.target_schema import (
    AuthBasic, AuthBearer, AuthHeader, AuthNone, AuthQuery,
    TargetConfig, TargetsFile,
)
from api.routes_targets import _redact_auth


# ---------------------------------------------------------------------------
# Variant construction
# ---------------------------------------------------------------------------
class TestAuthVariants:
    def test_none_default(self) -> None:
        a = AuthNone()
        assert a.type == "none"
        assert a.extra_headers == {}

    def test_bearer_with_env(self) -> None:
        a = AuthBearer(type="bearer", token_env="OPENAI_API_KEY")
        assert a.token_env == "OPENAI_API_KEY"
        assert a.token is None

    def test_bearer_with_inline_token(self) -> None:
        a = AuthBearer(type="bearer", token="sk-test")
        assert a.token == "sk-test"

    def test_bearer_with_extra_headers(self) -> None:
        a = AuthBearer(
            type="bearer",
            token_env="OPENAI_API_KEY",
            extra_headers={"OpenAI-Organization": "org-xyz"},
        )
        assert a.extra_headers["OpenAI-Organization"] == "org-xyz"

    def test_bearer_rejects_when_no_token_source(self) -> None:
        with pytest.raises(ValidationError):
            AuthBearer(type="bearer")

    def test_header_requires_non_empty_headers(self) -> None:
        with pytest.raises(ValidationError):
            AuthHeader(type="header", headers={})
        a = AuthHeader(type="header", headers={"X-API-Key": "v"})
        assert a.headers["X-API-Key"] == "v"

    def test_query_with_env(self) -> None:
        a = AuthQuery(type="query", query_key="key", query_value_env="GEMINI_KEY")
        assert a.query_key == "key"
        assert a.query_value_env == "GEMINI_KEY"

    def test_query_with_inline_value(self) -> None:
        a = AuthQuery(type="query", query_key="key", query_value="literal")
        assert a.query_value == "literal"

    def test_query_rejects_missing_value(self) -> None:
        with pytest.raises(ValidationError):
            AuthQuery(type="query", query_key="key")

    def test_query_rejects_empty_key(self) -> None:
        with pytest.raises(ValidationError):
            AuthQuery(type="query", query_key="", query_value_env="X")

    def test_basic_needs_user_and_password(self) -> None:
        with pytest.raises(ValidationError):
            AuthBasic(type="basic")
        with pytest.raises(ValidationError):
            AuthBasic(type="basic", username="alice")  # no password
        with pytest.raises(ValidationError):
            AuthBasic(type="basic", password_env="PW")  # no username
        a = AuthBasic(type="basic", username="alice", password_env="PW")
        assert a.username == "alice"
        assert a.password_env == "PW"


# ---------------------------------------------------------------------------
# TargetConfig discriminator + normalization
# ---------------------------------------------------------------------------
class TestTargetConfigAuthIntegration:
    def _api_base(self, **extra) -> dict:
        d = {"id": "t1", "name": "T", "type": "api", "endpoint": "https://x"}
        d.update(extra)
        return d

    def test_legacy_empty_dict_normalises_to_none(self) -> None:
        t = TargetConfig(**self._api_base(auth={}))
        assert t.auth.type == "none"

    def test_legacy_missing_auth_normalises_to_none(self) -> None:
        t = TargetConfig(**self._api_base())
        assert t.auth.type == "none"

    def test_legacy_dict_without_type_normalises_to_none(self) -> None:
        """token_env without `type` (older yaml) → AuthNone with extras dropped."""
        t = TargetConfig(**self._api_base(auth={"token_env": "X"}))
        assert t.auth.type == "none"

    def test_legacy_bearer_dict_round_trips(self) -> None:
        t = TargetConfig(**self._api_base(auth={"type": "bearer", "token_env": "FOO"}))
        assert t.auth.type == "bearer"
        assert t.auth.token_env == "FOO"

    def test_discriminator_picks_query_variant(self) -> None:
        t = TargetConfig(**self._api_base(auth={
            "type": "query", "query_key": "k", "query_value_env": "ENV",
        }))
        assert t.auth.type == "query"
        assert t.auth.query_key == "k"

    def test_extra_headers_carried_on_any_variant(self) -> None:
        for atype in ("none", "bearer"):
            spec = {"type": atype, "extra_headers": {"X-Tenant": "t1"}}
            if atype == "bearer":
                spec["token_env"] = "X"
            t = TargetConfig(**self._api_base(auth=spec))
            assert t.auth.extra_headers["X-Tenant"] == "t1"


# ---------------------------------------------------------------------------
# Redaction layer (defence-in-depth on every API read)
# ---------------------------------------------------------------------------
class TestRedactAuth:
    def test_redacts_bearer_token(self) -> None:
        out = _redact_auth({"auth": {"type": "bearer", "token": "secret123"}})
        assert out["auth"]["token"] == "***redacted***"

    def test_preserves_token_env_pointer(self) -> None:
        """`token_env` is a NAME, not a secret — must not be redacted."""
        out = _redact_auth({"auth": {"type": "bearer", "token_env": "OPENAI_API_KEY"}})
        assert out["auth"]["token_env"] == "OPENAI_API_KEY"

    def test_redacts_query_value(self) -> None:
        out = _redact_auth({"auth": {"type": "query", "query_key": "key",
                                       "query_value": "real-secret"}})
        assert out["auth"]["query_value"] == "***redacted***"
        assert out["auth"]["query_key"] == "key"  # name preserved

    def test_redacts_basic_password(self) -> None:
        out = _redact_auth({"auth": {"type": "basic", "username": "alice",
                                       "password": "hunter2"}})
        assert out["auth"]["password"] == "***redacted***"
        assert out["auth"]["username"] == "alice"

    def test_redacts_credential_headers(self) -> None:
        out = _redact_auth({"auth": {
            "type": "header",
            "headers": {
                "X-API-Key": "real-key",
                "Authorization": "Bearer xxx",
                "Content-Type": "application/json",  # not a secret
            },
        }})
        hdrs = out["auth"]["headers"]
        assert hdrs["X-API-Key"] == "***redacted***"
        assert hdrs["Authorization"] == "***redacted***"
        assert hdrs["Content-Type"] == "application/json"

    def test_empty_value_not_replaced(self) -> None:
        """Empty/falsy values aren't redacted — clearer in audit."""
        out = _redact_auth({"auth": {"type": "bearer", "token": ""}})
        assert out["auth"]["token"] == ""


# ---------------------------------------------------------------------------
# TargetsFile end-to-end: a yaml-shaped dict round-trips through everything
# ---------------------------------------------------------------------------
class TestTargetsFileRoundTrip:
    def test_mixed_auth_types_validate_together(self) -> None:
        data = {
            "version": 1,
            "targets": [
                {"id": "a", "name": "A", "type": "mock"},
                {"id": "b", "name": "B", "type": "api", "endpoint": "https://b",
                 "auth": {"type": "bearer", "token_env": "B_TOKEN"}},
                {"id": "c", "name": "C", "type": "api", "endpoint": "https://c",
                 "auth": {"type": "query", "query_key": "key", "query_value_env": "C_KEY"}},
                {"id": "d", "name": "D", "type": "api", "endpoint": "https://d",
                 "auth": {"type": "header", "headers": {"X-Api-Key": "literal"}}},
            ],
        }
        tf = TargetsFile(**data)
        types = [t.auth.type for t in tf.targets]
        assert types == ["none", "bearer", "query", "header"]
