"""api/routes_targets.py
External-eval target inventory + dashboard CRUD.

Reads + writes `external_eval/targets.yaml`. Writes go through the
validated helpers in `external_eval/target_loader.py`, which file-lock
read-modify-write so two dashboard tabs editing at once cannot leave
the file half-valid.

Auth secrets (`token`, `password`, `secret`, `api_key`) are redacted on
every read response. The dashboard cannot retrieve a token after it has
been stored — only re-set it.

Routes
------
GET    /targets               — list all targets (auth secrets redacted)
GET    /targets/{id}          — single target by id (auth secrets redacted)
POST   /targets               — upsert (insert or overwrite by id)
DELETE /targets/{id}          — remove by id
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from external_eval.target_loader import (
    delete_target,
    get_target,
    list_targets,
    upsert_target,
)
from schemas.target_schema import TargetConfig


router = APIRouter(prefix="/targets", tags=["targets"])


def _redact_auth(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Strip token / password fields from auth before returning to clients.

    The yaml stores `token_env` (an env-var name) rather than the secret
    itself, but defence-in-depth: never leak any field whose name suggests
    a credential.
    """
    auth = dict(payload.get("auth") or {})
    redacted_keys = {"token", "password", "secret", "api_key"}
    for k in list(auth.keys()):
        if k.lower() in redacted_keys:
            auth[k] = "***redacted***"
    out = dict(payload)
    out["auth"] = auth
    return out


def _serialise(target) -> Dict[str, Any]:
    """TargetConfig pydantic → redacted dict."""
    payload = target.model_dump(mode="json")
    return _redact_auth(payload)


@router.get("")
def list_all_targets(enabled_only: bool = False) -> Dict[str, Any]:
    try:
        targets = list_targets(enabled_only=enabled_only)
    except FileNotFoundError:
        return {"targets": [], "total": 0}
    rows: List[Dict[str, Any]] = [_serialise(t) for t in targets]
    return {"targets": rows, "total": len(rows)}


@router.get("/{target_id}")
def get_one_target(target_id: str) -> Dict[str, Any]:
    try:
        target = get_target(target_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="targets.yaml missing")
    if target is None:
        raise HTTPException(status_code=404, detail=f"target not found: {target_id}")
    return _serialise(target)


@router.post("")
def upsert_one_target(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Insert or overwrite a target. Validates against `TargetConfig`."""
    try:
        target = TargetConfig.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())
    upsert_target(target)
    return {"status": "ok", "id": target.id, "target": _serialise(target)}


@router.delete("/{target_id}")
def delete_one_target(target_id: str) -> Dict[str, Any]:
    """Remove a target by id. Returns 404 if not present."""
    try:
        removed = delete_target(target_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="targets.yaml missing")
    if not removed:
        raise HTTPException(status_code=404, detail=f"target not found: {target_id}")
    return {"status": "ok", "id": target_id}
