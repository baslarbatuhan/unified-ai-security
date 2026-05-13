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
POST   /targets/test          — dry-run probe of a candidate target (no save)
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
    """Strip secret fields from auth before returning to clients.

    The yaml is supposed to use `*_env` variants (env-var names, safe to
    show) but defence-in-depth — never leak any field whose name suggests
    a credential, regardless of auth.type. Hafta 11.2 expands the field
    list to cover the new `query_value` (Gemini-style) and `password`
    variants alongside the legacy `token`.
    """
    auth = dict(payload.get("auth") or {})

    # Top-level direct-value fields that must never leave the server.
    direct_secret_fields = {
        "token", "password", "secret", "api_key", "query_value",
    }
    for k in list(auth.keys()):
        if k.lower() in direct_secret_fields and auth[k]:
            auth[k] = "***redacted***"

    # `header` auth stores arbitrary headers; the value of any header
    # whose name looks credential-y must be redacted too. Conservative
    # name match — better to over-redact than leak.
    if isinstance(auth.get("headers"), dict):
        hdrs = dict(auth["headers"])
        for hk, hv in list(hdrs.items()):
            low = str(hk).lower()
            if any(s in low for s in ("auth", "token", "key", "secret", "password")):
                if hv:
                    hdrs[hk] = "***redacted***"
        auth["headers"] = hdrs

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


# ---------------------------------------------------------------------------
# POST /targets/test — dry-run a candidate target without persisting.
# ---------------------------------------------------------------------------
def _categorise_error(error_class: str | None, error_message: str | None) -> str:
    """Map adapter error classes onto the dashboard's 5-bucket palette.
    Used by the UI to colour-code the inline result without re-parsing
    the error string client-side."""
    cls = (error_class or "").lower()
    msg = (error_message or "").lower()
    if cls == "timeout" or "timed out" in msg or "timeout" in msg:
        return "timeout"
    if cls in ("adaptertransporterror",) or "http " in msg or "transport" in msg:
        return "transport"
    if cls in ("adapterconfigerror",):
        return "config"
    if "validation" in msg or "schema" in msg:
        return "schema"
    if cls and cls != "unexpected":
        return cls
    return "unexpected" if cls else ""


@router.post("/test")
def test_target_connection(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send a single probe prompt to a candidate target without writing it.

    Body shape:
        {
          "target": { ... full TargetConfig dict, just like POST / ... },
          "probe_prompt": "ping"   # optional, default "ping"
        }

    The target dict is validated against `TargetConfig` first; if that
    fails we return ok=false with category="schema" so the dashboard can
    show the field error inline. Otherwise we build the matching adapter,
    send one prompt, capture the `ChatbotResponse`, and close the adapter.
    The adapter itself never raises — `ChatbotAdapter.send` already wraps
    every exception into `ChatbotResponse(ok=False, …)`. We simply expose
    that to the caller plus an error category for colour-coding.
    """
    target_payload = payload.get("target")
    if not isinstance(target_payload, dict):
        raise HTTPException(
            status_code=422,
            detail="`target` field is required and must be an object",
        )
    probe = str(payload.get("probe_prompt") or "ping")

    # Schema validation — surface field errors as ok=false (not 422) so the
    # dashboard treats this like any other adapter failure (single inline
    # banner) instead of crashing the request.
    try:
        target = TargetConfig.model_validate(target_payload)
    except ValidationError as exc:
        # Pydantic's `errors()` includes a `ctx` with the original
        # exception object on model_validator failures, which is not
        # JSON-serialisable. Strip down to the four scalar fields.
        raw_errors = exc.errors()
        safe_errors = [
            {
                "loc": list(e.get("loc", ())),
                "msg": str(e.get("msg", "")),
                "type": str(e.get("type", "")),
                "input": str(e.get("input", ""))[:120],
            }
            for e in raw_errors
        ]
        return {
            "ok": False,
            "category": "schema",
            "error_message": safe_errors[0]["msg"] if safe_errors else str(exc),
            "error_details": safe_errors,
            "target_id": target_payload.get("id", ""),
        }

    # Lazy import — adapter_factory pulls httpx/playwright on demand,
    # don't penalise the unrelated GET /targets paths.
    from external_eval.adapter_factory import build_adapter
    from external_eval.base_adapter import AdapterConfigError

    try:
        adapter = build_adapter(target)
    except AdapterConfigError as exc:
        return {
            "ok": False,
            "category": "config",
            "error_message": str(exc),
            "target_id": target.id,
        }

    try:
        response = adapter.send(probe)
    finally:
        try:
            adapter.close()
        except Exception:
            pass  # close() best-effort — never mask the original error

    if response.ok:
        sample = response.text or ""
        if len(sample) > 200:
            sample = sample[:200] + "…"
        return {
            "ok": True,
            "category": "",
            "target_id": target.id,
            "probe_prompt": probe,
            "latency_ms": response.latency_ms,
            "response_sample": sample,
            "response_chars": len(response.text or ""),
            "metadata": response.metadata,
        }
    else:
        return {
            "ok": False,
            "category": _categorise_error(
                (response.metadata or {}).get("error_class"),
                response.error_message,
            ),
            "target_id": target.id,
            "probe_prompt": probe,
            "latency_ms": response.latency_ms,
            "error_message": response.error_message or "unknown error",
            "metadata": response.metadata,
        }
