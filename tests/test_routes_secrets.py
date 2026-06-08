"""tests/test_routes_secrets.py
=================================
Integration tests for the gateway's ``/secrets`` router.

We mount the router on a bare FastAPI app, point the vault file at a
``tmp_path``, and exercise the CRUD surface end-to-end through
``TestClient``. No real network traffic — we just verify that:

* The router never leaks values back to the client.
* Source diagnostics ('env' / 'vault' / 'missing') reflect reality.
* PUT writes round-trip and DELETE returns 404 when nothing was removed.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    """Vault isolated under tmp_path; fresh router on each test."""
    vault = tmp_path / "secrets.yaml"
    monkeypatch.setenv("UAIS_SECRETS_PATH", str(vault))

    # Reimport in case prior tests cached module-level state. The router
    # is stateless (it just calls into secrets_store) so a single import
    # is fine in practice, but explicit beats fragile.
    from api.routes_secrets import router as secrets_router

    app = FastAPI()
    app.include_router(secrets_router)
    return TestClient(app)


class TestListAndGet:
    def test_list_empty_vault(self, client: TestClient) -> None:
        r = client.get("/secrets")
        assert r.status_code == 200
        body = r.json()
        assert body == {"names": [], "items": []}

    def test_list_after_set(self, client: TestClient) -> None:
        client.put("/secrets/ALPHA", json={"value": "a"})
        client.put("/secrets/BRAVO", json={"value": "b"})

        r = client.get("/secrets")
        assert r.status_code == 200
        body = r.json()
        assert body["names"] == ["ALPHA", "BRAVO"]
        sources = {it["name"]: it["source"] for it in body["items"]}
        assert sources == {"ALPHA": "vault", "BRAVO": "vault"}
        # No "value" key may leak in the items list.
        for it in body["items"]:
            assert "value" not in it

    def test_get_missing_name_is_200_with_missing_source(self, client: TestClient) -> None:
        r = client.get("/secrets/NEVER_SET")
        assert r.status_code == 200
        assert r.json() == {"name": "NEVER_SET", "source": "missing", "in_vault": False}

    def test_get_reflects_env_when_set(self, client: TestClient, monkeypatch) -> None:
        monkeypatch.setenv("FROM_ENV_ONLY", "x")
        r = client.get("/secrets/FROM_ENV_ONLY")
        assert r.json() == {"name": "FROM_ENV_ONLY", "source": "env", "in_vault": False}

    def test_env_shadows_vault_in_source(self, client: TestClient, monkeypatch) -> None:
        client.put("/secrets/SHARED", json={"value": "vault-side"})
        monkeypatch.setenv("SHARED", "env-side")
        r = client.get("/secrets/SHARED")
        body = r.json()
        assert body["source"] == "env"
        # ...but the vault still has its entry.
        assert body["in_vault"] is True


class TestUpsertAndDelete:
    def test_put_creates_then_get_confirms(self, client: TestClient) -> None:
        r = client.put("/secrets/MY_KEY", json={"value": "shh"})
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "MY_KEY"
        assert body["source"] == "vault"
        assert body["in_vault"] is True
        # The value never echoes back.
        assert "value" not in body

        # Round-trip via GET.
        assert client.get("/secrets/MY_KEY").json()["source"] == "vault"

    def test_put_overwrites(self, client: TestClient) -> None:
        client.put("/secrets/K", json={"value": "v1"})
        client.put("/secrets/K", json={"value": "v2"})
        # We can't observe the value through the API; verify via the
        # store directly. (This is what the api_adapter would see.)
        from external_eval import secrets_store
        assert secrets_store.get("K") == "v2"

    def test_put_empty_value_allowed(self, client: TestClient) -> None:
        r = client.put("/secrets/BLANK", json={"value": ""})
        assert r.status_code == 200
        # Source is 'missing' because a blank value isn't useful — see
        # secret_resolver.source_of for the rationale.
        assert r.json()["source"] == "missing"
        assert r.json()["in_vault"] is True

    def test_put_empty_name_rejected(self, client: TestClient) -> None:
        # FastAPI routes the path /secrets/ to the list endpoint (which
        # is PUT-less) — verify a whitespace-only name is rejected by
        # the handler when supplied.
        r = client.put("/secrets/%20", json={"value": "x"})
        assert r.status_code == 400

    def test_delete_removes(self, client: TestClient) -> None:
        client.put("/secrets/GONE", json={"value": "bye"})
        r = client.delete("/secrets/GONE")
        assert r.status_code == 200
        body = r.json()
        assert body["in_vault"] is False
        assert body["source"] == "missing"

    def test_delete_missing_is_404(self, client: TestClient) -> None:
        r = client.delete("/secrets/NEVER")
        assert r.status_code == 404


class TestNoValueLeakage:
    """Defence-in-depth: no GET path should ever surface a stored value."""

    def test_list_response_excludes_values(self, client: TestClient) -> None:
        client.put("/secrets/SHHH", json={"value": "leaked?"})
        r = client.get("/secrets")
        text = r.text
        assert "leaked?" not in text

    def test_get_single_excludes_value(self, client: TestClient) -> None:
        client.put("/secrets/SHHH", json={"value": "leaked?"})
        r = client.get("/secrets/SHHH")
        assert "leaked?" not in r.text

    def test_delete_response_excludes_value(self, client: TestClient) -> None:
        client.put("/secrets/SHHH", json={"value": "leaked?"})
        r = client.delete("/secrets/SHHH")
        assert "leaked?" not in r.text
