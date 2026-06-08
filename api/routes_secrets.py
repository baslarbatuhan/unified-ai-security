"""api/routes_secrets.py
============================
Gateway endpoints for the local secrets vault
(``external_eval/secrets_store``).

The dashboard's Targets page POSTs a pasted value here when the user
adds or edits a target; the api_adapter later resolves the value at
request time via ``secret_resolver.resolve(name)`` (env-var first,
vault as fallback).

Security notes
--------------
* Values **never** leave the gateway via these routes. ``GET /secrets``
  returns names + source diagnostics only.
* Each route logs only the secret *name*, not its value. (Default FastAPI
  access logs do not log request bodies, but we double-check by never
  reflecting the body back in error messages either.)
* The vault file lives under ``runs/`` which is git-ignored and
  bind-mounted in the gateway container; no value is ever committed.

Routes
------
GET    /secrets               — list known names + per-name source (env/vault/missing)
GET    /secrets/{name}        — single-name source lookup (still no value)
PUT    /secrets/{name}        — upsert; body ``{"value": "..."}``
DELETE /secrets/{name}        — remove; 404 if not present
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from external_eval import secrets_store
from external_eval.secret_resolver import source_of


router = APIRouter(prefix="/secrets", tags=["secrets"])


class SecretValue(BaseModel):
    value: str = Field(
        ...,
        description=(
            "Secret value to store. Empty string is allowed (acts as "
            "an explicit blank entry) — to remove the entry entirely "
            "use DELETE."
        ),
    )


class SecretInfo(BaseModel):
    name: str
    source: str = Field(
        ...,
        description=(
            "Where the value would be resolved from RIGHT NOW: "
            "'env' (os.environ has it; vault is shadowed), "
            "'vault' (resolved from the local vault), "
            "'missing' (neither env nor vault has a non-empty value)."
        ),
    )
    in_vault: bool = Field(
        ...,
        description="Whether the local vault has an entry for this name.",
    )


class SecretsListResponse(BaseModel):
    names: List[str]
    items: List[SecretInfo]


def _info(name: str) -> SecretInfo:
    return SecretInfo(
        name=name,
        source=source_of(name),
        in_vault=secrets_store.has(name),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.get("", response_model=SecretsListResponse)
def list_secrets() -> SecretsListResponse:
    """Return every name known to the vault plus its current source.

    The dashboard uses this to render "🔒 Stored" / "🌱 from env-var"
    badges next to each ``*_env`` field on the Targets form.
    """
    names = secrets_store.list_names()
    return SecretsListResponse(
        names=names,
        items=[_info(n) for n in names],
    )


@router.get("/{name}", response_model=SecretInfo)
def get_secret(name: str) -> SecretInfo:
    """Look up a single name without revealing its value.

    Returns a 200 for both vault-present and vault-absent names — the
    ``source`` field tells the caller which case it is. (404 would
    leak whether a name exists; we don't keep names private but we
    do keep the API uniform.)
    """
    return _info(name)


@router.put("/{name}", response_model=SecretInfo)
def upsert_secret(name: str, payload: SecretValue) -> SecretInfo:
    """Insert or overwrite the vault entry for ``name``.

    Empty body value is accepted (records an explicit blank). To remove
    an entry entirely use ``DELETE /secrets/{name}``.
    """
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="secret name cannot be empty")
    secrets_store.set(name, payload.value)
    return _info(name)


@router.delete("/{name}", response_model=SecretInfo)
def delete_secret(name: str) -> SecretInfo:
    """Remove ``name`` from the vault. 404 if the name was not present.

    Returns the post-delete source state so the dashboard can refresh
    the badge in one round-trip (e.g. env-var still set → source='env').
    """
    removed = secrets_store.delete(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"no vault entry named {name!r}")
    return _info(name)
