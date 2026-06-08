"""tests/test_secret_resolver.py
=================================
Unit tests for ``external_eval.secret_resolver`` — the helper that
api_adapter uses to look up an ``auth.*_env`` reference, with env-var
priority over the local vault.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from external_eval import secrets_store
from external_eval.secret_resolver import resolve, source_of


@pytest.fixture
def vault(tmp_path: Path, monkeypatch) -> Path:
    p = tmp_path / "secrets.yaml"
    monkeypatch.setenv("UAIS_SECRETS_PATH", str(p))
    return p


class TestResolve:
    def test_empty_name_returns_empty(self, vault: Path) -> None:
        assert resolve("") == ""

    def test_env_wins_over_vault(self, vault: Path, monkeypatch) -> None:
        secrets_store.set("KEY", "from-vault")
        monkeypatch.setenv("KEY", "from-env")
        assert resolve("KEY") == "from-env"

    def test_vault_used_when_env_missing(self, vault: Path, monkeypatch) -> None:
        monkeypatch.delenv("KEY", raising=False)
        secrets_store.set("KEY", "from-vault")
        assert resolve("KEY") == "from-vault"

    def test_empty_env_falls_through_to_vault(self, vault: Path, monkeypatch) -> None:
        """An empty env-var (``KEY=`` in .env) must not shadow the vault.

        This is the bootstrap-friendliness invariant — users can leave a
        commented-out env-var in .env and still have the vault entry win.
        """
        monkeypatch.setenv("KEY", "")
        secrets_store.set("KEY", "from-vault")
        assert resolve("KEY") == "from-vault"

    def test_missing_everywhere_returns_empty(self, vault: Path, monkeypatch) -> None:
        monkeypatch.delenv("KEY", raising=False)
        assert resolve("KEY") == ""


class TestSourceOf:
    def test_missing_when_neither_set(self, vault: Path, monkeypatch) -> None:
        monkeypatch.delenv("KEY", raising=False)
        assert source_of("KEY") == "missing"

    def test_env_when_only_env_set(self, vault: Path, monkeypatch) -> None:
        monkeypatch.setenv("KEY", "x")
        assert source_of("KEY") == "env"

    def test_vault_when_only_vault_set(self, vault: Path, monkeypatch) -> None:
        monkeypatch.delenv("KEY", raising=False)
        secrets_store.set("KEY", "x")
        assert source_of("KEY") == "vault"

    def test_env_wins_in_diagnostics_too(self, vault: Path, monkeypatch) -> None:
        monkeypatch.setenv("KEY", "x")
        secrets_store.set("KEY", "y")
        assert source_of("KEY") == "env"

    def test_empty_vault_value_is_missing(self, vault: Path, monkeypatch) -> None:
        """Blank vault entry isn't useful — diagnostics should report 'missing'
        so the dashboard surfaces the warning instead of a false 'stored' badge."""
        monkeypatch.delenv("KEY", raising=False)
        secrets_store.set("KEY", "")
        assert source_of("KEY") == "missing"

    def test_empty_name_is_missing(self, vault: Path) -> None:
        assert source_of("") == "missing"
