"""tests/test_secrets_store.py
================================
Unit tests for ``external_eval.secrets_store`` — the local YAML vault
that backs dashboard-pasted credentials.

The store reads its path from ``UAIS_SECRETS_PATH`` (with a default
under ``runs/.secrets.yaml``); these tests monkey-patch the env-var to
a ``tmp_path`` so each test gets an isolated file.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from external_eval import secrets_store


@pytest.fixture
def vault(tmp_path: Path, monkeypatch) -> Path:
    """Isolate the vault file under tmp_path for one test."""
    p = tmp_path / "secrets.yaml"
    monkeypatch.setenv("UAIS_SECRETS_PATH", str(p))
    return p


class TestCrud:
    def test_missing_file_is_empty(self, vault: Path) -> None:
        assert not vault.exists()
        assert secrets_store.get("ANY") is None
        assert secrets_store.list_names() == []
        assert secrets_store.has("ANY") is False

    def test_set_then_get(self, vault: Path) -> None:
        secrets_store.set("MY_KEY", "shh")
        assert vault.exists()
        assert secrets_store.get("MY_KEY") == "shh"
        assert secrets_store.has("MY_KEY") is True

    def test_set_overwrites(self, vault: Path) -> None:
        secrets_store.set("K", "v1")
        secrets_store.set("K", "v2")
        assert secrets_store.get("K") == "v2"

    def test_empty_value_allowed(self, vault: Path) -> None:
        """Empty string is a legitimate value (e.g. user cleared a field)."""
        secrets_store.set("K", "")
        assert secrets_store.has("K") is True
        assert secrets_store.get("K") == ""

    def test_delete_removes(self, vault: Path) -> None:
        secrets_store.set("K", "v")
        assert secrets_store.delete("K") is True
        assert secrets_store.has("K") is False
        assert secrets_store.delete("K") is False  # idempotent on second call

    def test_list_names_sorted(self, vault: Path) -> None:
        for n in ("ZED", "ALPHA", "MIKE"):
            secrets_store.set(n, "x")
        assert secrets_store.list_names() == ["ALPHA", "MIKE", "ZED"]

    def test_empty_name_rejected(self, vault: Path) -> None:
        with pytest.raises(ValueError):
            secrets_store.set("", "v")

    def test_get_empty_name(self, vault: Path) -> None:
        """Empty-name lookup returns None, not an exception."""
        assert secrets_store.get("") is None


class TestPersistence:
    def test_round_trip_on_disk(self, vault: Path) -> None:
        secrets_store.set("ROUND", "trip")
        raw = yaml.safe_load(vault.read_text(encoding="utf-8"))
        assert raw == {"version": 1, "secrets": {"ROUND": "trip"}}

    def test_unicode_value_preserved(self, vault: Path) -> None:
        secrets_store.set("U", "değer-Ω-😀")
        assert secrets_store.get("U") == "değer-Ω-😀"

    def test_garbage_payload_treated_as_empty(self, vault: Path) -> None:
        """A YAML file that parses but isn't shaped like ``{secrets: {...}}``
        must degrade gracefully — callers can recover by overwriting via
        ``set()``. We don't try to *catch* hard YAMLError here; that
        would mask actual filesystem corruption from operators."""
        # A plain scalar parses cleanly into a string, exercising the
        # "doc isn't a dict" branch in _load_raw.
        vault.write_text("just a scalar string", encoding="utf-8")
        assert secrets_store.list_names() == []
        assert secrets_store.get("ANY") is None
        # And we can recover by writing.
        secrets_store.set("RECOVERED", "ok")
        assert secrets_store.get("RECOVERED") == "ok"

    def test_wrong_shape_yaml_returns_empty(self, vault: Path) -> None:
        """A YAML file that parses but has no ``secrets:`` map → empty."""
        vault.write_text("version: 1\nother: stuff\n", encoding="utf-8")
        assert secrets_store.list_names() == []
        assert secrets_store.get("ANY") is None

    def test_file_permissions_owner_only(self, vault: Path) -> None:
        """POSIX-only smoke: vault file is mode 0600 after write."""
        if os.name != "posix":
            pytest.skip("permissions check is POSIX-only")
        secrets_store.set("K", "v")
        mode = vault.stat().st_mode & 0o777
        assert mode == 0o600
