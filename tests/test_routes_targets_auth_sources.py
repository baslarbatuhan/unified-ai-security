"""tests/test_routes_targets_auth_sources.py
==============================================
Sprint 3.3 — ``/targets/test`` exposes an ``auth_sources`` map showing
where each secret reference would resolve from RIGHT NOW (env / vault /
missing).  Values are NEVER returned — only the env-var name and the
diagnostic source label.

We unit-test the helper directly (cheap and exhaustive) and rely on the
broader route tests in test_phase5_routes.py for end-to-end coverage.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from api.routes_targets import _collect_auth_sources
from external_eval import secrets_store
from schemas.target_schema import TargetConfig


@pytest.fixture
def vault(tmp_path: Path, monkeypatch) -> Path:
    """Isolate the vault file so source diagnostics are deterministic."""
    p = tmp_path / "secrets.yaml"
    monkeypatch.setenv("UAIS_SECRETS_PATH", str(p))
    return p


def _t(**auth) -> TargetConfig:
    return TargetConfig(
        id="t", name="T", type="api", endpoint="https://x.test/c",
        auth=auth,
    )


class TestCollectAuthSources:
    def test_none_auth_with_no_extras_returns_empty(self, vault: Path) -> None:
        assert _collect_auth_sources(_t(type="none")) == {}

    def test_bearer_env_only(self, vault: Path, monkeypatch) -> None:
        monkeypatch.setenv("MY_BEARER", "x")
        out = _collect_auth_sources(_t(type="bearer", token_env="MY_BEARER"))
        assert out == {"auth.token_env: MY_BEARER": "env"}

    def test_bearer_vault_only(self, vault: Path, monkeypatch) -> None:
        monkeypatch.delenv("MY_BEARER", raising=False)
        secrets_store.set("MY_BEARER", "x")
        out = _collect_auth_sources(_t(type="bearer", token_env="MY_BEARER"))
        assert out == {"auth.token_env: MY_BEARER": "vault"}

    def test_bearer_missing(self, vault: Path, monkeypatch) -> None:
        monkeypatch.delenv("ABSENT", raising=False)
        out = _collect_auth_sources(_t(type="bearer", token_env="ABSENT"))
        assert out == {"auth.token_env: ABSENT": "missing"}

    def test_bearer_inline_token_does_not_create_ref(self, vault: Path) -> None:
        """Inline tokens are legacy. No env-var to diagnose → no entry."""
        out = _collect_auth_sources(_t(type="bearer", token="literal"))
        assert out == {}

    def test_query_env(self, vault: Path, monkeypatch) -> None:
        monkeypatch.setenv("QK", "v")
        out = _collect_auth_sources(_t(
            type="query", query_key="key", query_value_env="QK",
        ))
        assert out == {"auth.query_value_env: QK": "env"}

    def test_basic_both_credentials(self, vault: Path, monkeypatch) -> None:
        monkeypatch.setenv("BU", "u")
        monkeypatch.delenv("BP", raising=False)
        secrets_store.set("BP", "p")
        out = _collect_auth_sources(_t(
            type="basic",
            username_env="BU", password_env="BP",
        ))
        assert out == {
            "auth.username_env: BU": "env",
            "auth.password_env: BP": "vault",
        }

    def test_header_scans_value_placeholders(self, vault: Path, monkeypatch) -> None:
        monkeypatch.setenv("HK", "k")
        monkeypatch.delenv("HM", raising=False)
        out = _collect_auth_sources(_t(
            type="header",
            headers={
                "x-api-key": "${HK}",
                "x-org": "literal-no-var",
                "x-fallback": "${HM}",
            },
        ))
        assert out == {
            "auth.headers ${HK}": "env",
            "auth.headers ${HM}": "missing",
        }

    def test_extra_headers_also_scanned(self, vault: Path, monkeypatch) -> None:
        """``extra_headers`` may carry ``${VAR}`` regardless of auth.type
        (Sprint 1 made interpolation symmetric)."""
        monkeypatch.setenv("EXTRA_KEY", "v")
        out = _collect_auth_sources(_t(
            type="none",
            extra_headers={"X-Inject": "${EXTRA_KEY}"},
        ))
        assert out == {"auth.extra_headers ${EXTRA_KEY}": "v" and "env"}

    def test_no_value_leakage(self, vault: Path, monkeypatch) -> None:
        """The actual secret value must never appear in the diagnostics."""
        secret_value = "super-secret-do-not-leak-12345"
        monkeypatch.setenv("LEAKY", secret_value)
        out = _collect_auth_sources(_t(type="bearer", token_env="LEAKY"))
        # Every key + value in the result is name/label/source — none of
        # them should contain the actual secret.
        flat = " ".join(list(out.keys()) + list(out.values()))
        assert secret_value not in flat
