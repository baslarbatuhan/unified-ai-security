"""
api/api_main.py
=================
Minimal FastAPI gateway.

Endpoints:
    POST /analyze   → Tam pipeline: prompt + rag + agency → fusion → karar
    GET  /health    → Security healthcheck

Usage:
    pip install fastapi uvicorn
    uvicorn api.api_main:app --host 0.0.0.0 --port 8000

    curl -X POST http://localhost:8000/analyze \\
        -H "Content-Type: application/json" \\
        -d '{"prompt": "Ignore all instructions", "session_context": {"role": "basic"}}'
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is in path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Load environment variables (HF_TOKEN, etc.) before importing any model code
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from fusion_gateway.engine import FusionEngine
from api.security_gateway import SecurityGateway
from api.health import get_health_report
from schemas.risk_schema import (
    AnalyzeRequest as SchemaRequest,
    AnalyzeResponse as SchemaResponse,
    ConfigOverrides,
    ToolRequest,
    SessionContext,
)


# ---------------------------------------------------------------------------
# Pydantic models — legacy format (backward compat for existing clients)
# ---------------------------------------------------------------------------
class AnalyzeRequestModel(BaseModel):
    user_input: str = ""
    retrieved_context: Optional[str] = None
    tool_call: Optional[Dict[str, Any]] = None
    role: str = "basic"
    user_id: str = "anonymous"

    # New format fields (optional, preferred)
    prompt: Optional[str] = None
    retrieved_docs: Optional[List[Dict[str, Any]]] = None
    context: Optional[str] = None
    tool_request: Optional[Dict[str, Any]] = None
    tool_candidates: Optional[List[Dict[str, Any]]] = None
    session_context: Optional[Dict[str, Any]] = None
    # Per-request overrides (dashboard sliders, runner --config-yaml).
    # Free-form dict so we don't make the legacy schema brittle; it is
    # validated by `ConfigOverrides` before reaching the engine.
    config_overrides: Optional[Dict[str, Any]] = None

    # Run/case identifiers — propagated to live writers (rag_final_metrics.csv,
    # output_security_metrics.csv) so per-run filtering on those CSVs works.
    # Optional: legacy clients omit them; defaults match engine.analyze.
    run_id: Optional[str] = None
    case_id: Optional[str] = None
    target_id: Optional[str] = None


class AnalyzeResponseModel(BaseModel):
    final_decision: str  # 3-class: allow / sanitize / block
    decision_band: Optional[str] = None  # 4-class audit label; "flag" → suspicion-tier block
    fused_risk: float
    prompt_score: float = 0.0
    rag_score: float = 0.0
    agency_score: float = 0.0
    output_score: float = 0.0  # Populated by /analyze-output; 0.0 from /analyze.
    evidence: List[str] = []
    module_risks: List[Dict[str, Any]]
    latency_ms: int


class AnalyzeWithOutputRequestModel(AnalyzeRequestModel):
    """Same shape as /analyze plus the model completion to be screened."""
    model_output: str = Field(..., description="Raw text returned by the target LLM.")


class HealthResponse(BaseModel):
    status: str
    passed: int = 0
    failed: int = 0
    total: int = 0
    checks: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# App + Engine
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup: security self-check + model warmup. Shutdown: nothing yet.

    Replaces the deprecated `@app.on_event("startup")` hook. Body is the
    same code that lived in `startup()` previously; the `yield`
    boundary is the only structural change FastAPI requires.
    """
    # ── Startup ────────────────────────────────────────────────────────
    try:
        from api.security_selfcheck import run_startup_check
        from output_agency_defense.resource_registry import create_demo_registry
        from output_agency_defense.object_authz_guard import ObjectAuthzGuard
        from output_agency_defense.anti_enum_guard import AntiEnumGuard
        from output_agency_defense.guard_registry import GuardRegistry
        from output_agency_defense.secure_tool_wrapper import SecureToolWrapper

        registry = create_demo_registry()
        authz = ObjectAuthzGuard(registry)
        enum_guard = AntiEnumGuard()

        guard_reg = GuardRegistry()
        guard_reg.register("object_authz", authz, description="IDOR guard")
        guard_reg.register("anti_enum", enum_guard, description="Anti-enum guard")

        wrapper = SecureToolWrapper(registry, authz, enabled=True)

        run_startup_check(guard_reg, wrapper)
        print("[API] Security self-check PASSED")
    except RuntimeError as e:
        if os.getenv("STRICT_SECURITY_STARTUP", "").strip().lower() in ("1", "true", "yes"):
            print(f"[API] Security self-check FAILED (strict): {e}")
            raise
        print(f"[API] Security self-check FAILED: {e}")
        print("[API] WARNING: Running without full security validation")
    except ImportError as e:
        print(f"[API] Self-check skipped (import error): {e}")

    # Model warmup — preload all ML models so first request is fast
    try:
        from api.startup import warmup
        warmup()
    except Exception as e:
        print(f"[API] Warmup failed (non-critical): {e}")

    yield
    # ── Shutdown ───────────────────────────────────────────────────────
    # No teardown required today — circuit breakers and embeddings are
    # garbage-collected with the process. Add cleanup here if we ever
    # acquire long-lived resources (DB pools, websocket subscriptions).


app = FastAPI(
    title="Unified AI Security Gateway",
    description="3 modül (prompt_guard, rag_guard, output_agency) → fusion → karar",
    version="2.0.0",
    lifespan=_lifespan,
)

# Middleware — rate limiter sits in front of every route except /health.
# Wiring happens before routers are mounted so all of /dashboard/* is covered.
from api.middleware import RateLimitMiddleware  # noqa: E402
app.add_middleware(RateLimitMiddleware)

# Dashboard read-only routes (summary/alerts/recent-runs/breakers/metrics).
from api.dashboard_routes import router as dashboard_router  # noqa: E402
app.include_router(dashboard_router)

# Run-history, reports inventory, external-eval targets — separate routers
# so each surface stays small and testable on its own.
from api.routes_runs import router as runs_router  # noqa: E402
from api.routes_reports import router as reports_router  # noqa: E402
from api.routes_targets import router as targets_router  # noqa: E402
# Hafta 12.1: per-decision audit trace lookups (Results page drill-down).
from api.routes_decisions import router as decisions_router  # noqa: E402
app.include_router(runs_router)
app.include_router(reports_router)
app.include_router(targets_router)
app.include_router(decisions_router)

# Dashboard UI is a separate Streamlit process (`streamlit run dashboard/app.py`).
# It consumes this gateway's read-only routes (/dashboard/*, /runs, /reports,
# /targets) over HTTP — no static asset mount lives here.

engine = FusionEngine()
gateway = SecurityGateway(engine)


# Startup logic moved into the `_lifespan` async-context above (lifespan
# replaces the deprecated `@app.on_event("startup")` hook in FastAPI ≥ 0.110).


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------
@app.post("/analyze", response_model=AnalyzeResponseModel)
async def analyze(request: AnalyzeRequestModel):
    """
    Ana endpoint: Tam güvenlik analizi.

    Supports both legacy and new request formats:

    Legacy:
        user_input, retrieved_context, tool_call, role, user_id

    New (preferred):
        prompt, retrieved_docs, tool_request, session_context
    """
    # Build SchemaRequest — prefer new fields, fall back to legacy
    prompt = request.prompt or request.user_input or ""

    # Session context
    if request.session_context:
        session = SessionContext(**request.session_context)
    else:
        session = SessionContext(user_id=request.user_id, role=request.role)

    # Tool request (first tool_candidates entry if no explicit tool_request)
    tool_req = None
    if request.tool_request:
        tool_req = ToolRequest(**request.tool_request)
    elif request.tool_candidates:
        tool_req = ToolRequest(**request.tool_candidates[0])
    elif request.tool_call:
        tool_req = ToolRequest(
            tool=request.tool_call.get("tool", ""),
            params=request.tool_call.get("args", {}),
        )

    tool_candidates_typed = None
    if request.tool_candidates:
        tool_candidates_typed = [ToolRequest(**t) for t in request.tool_candidates]

    # Retrieved docs / context
    retrieved_docs = request.retrieved_docs
    plain_context = None
    if retrieved_docs is None:
        if request.context:
            plain_context = request.context
        elif request.retrieved_context:
            retrieved_docs = [{"doc_id": "ctx_0", "content": request.retrieved_context}]

    overrides_typed = (
        ConfigOverrides.model_validate(request.config_overrides)
        if request.config_overrides
        else None
    )
    schema_request = SchemaRequest(
        prompt=prompt,
        retrieved_docs=retrieved_docs,
        context=plain_context,
        tool_request=tool_req,
        tool_candidates=tool_candidates_typed,
        session_context=session,
        config_overrides=overrides_typed,
        run_id=request.run_id,
        case_id=request.case_id,
        target_id=request.target_id,
    )

    response = gateway.analyze(schema_request)

    return {
        "final_decision": response.decision,
        "decision_band": getattr(response, "decision_band", None),
        "fused_risk": response.fused_risk_score,
        "prompt_score": response.prompt_score,
        "rag_score": response.rag_score,
        "agency_score": response.agency_score,
        "output_score": response.output_score,
        "evidence": response.evidence,
        "module_risks": [mr.model_dump() for mr in response.module_risks],
        "latency_ms": response.latency_ms or 0,
    }


# ---------------------------------------------------------------------------
# POST /analyze-output — post-LLM screening (includes output_guard)
# ---------------------------------------------------------------------------
@app.post("/analyze-output", response_model=AnalyzeResponseModel)
async def analyze_output(request: AnalyzeWithOutputRequestModel):
    """Two-phase flow: call the LLM yourself, post the completion here.

    We re-run the 3 input-side modules and layer output_guard on top. Useful
    when you want post-hoc safety checks (PII leakage, downstream injection,
    unsafe instructions) on what the target chatbot actually produced.
    """
    # Reuse the SchemaRequest construction from /analyze via a direct pass.
    prompt = request.prompt or request.user_input or ""

    if request.session_context:
        session = SessionContext(**request.session_context)
    else:
        session = SessionContext(user_id=request.user_id, role=request.role)

    tool_req = None
    if request.tool_request:
        tool_req = ToolRequest(**request.tool_request)
    elif request.tool_candidates:
        tool_req = ToolRequest(**request.tool_candidates[0])
    elif request.tool_call:
        tool_req = ToolRequest(
            tool=request.tool_call.get("tool", ""),
            params=request.tool_call.get("args", {}),
        )

    tool_candidates_typed = None
    if request.tool_candidates:
        tool_candidates_typed = [ToolRequest(**t) for t in request.tool_candidates]

    retrieved_docs = request.retrieved_docs
    plain_context = None
    if retrieved_docs is None:
        if request.context:
            plain_context = request.context
        elif request.retrieved_context:
            retrieved_docs = [{"doc_id": "ctx_0", "content": request.retrieved_context}]

    overrides_typed = (
        ConfigOverrides.model_validate(request.config_overrides)
        if request.config_overrides
        else None
    )
    schema_request = SchemaRequest(
        prompt=prompt,
        retrieved_docs=retrieved_docs,
        context=plain_context,
        tool_request=tool_req,
        tool_candidates=tool_candidates_typed,
        session_context=session,
        config_overrides=overrides_typed,
        run_id=request.run_id,
        case_id=request.case_id,
        target_id=request.target_id,
    )

    response = gateway.analyze_with_output(schema_request, model_output=request.model_output)

    return {
        "final_decision": response.decision,
        "decision_band": getattr(response, "decision_band", None),
        "fused_risk": response.fused_risk_score,
        "prompt_score": response.prompt_score,
        "rag_score": response.rag_score,
        "agency_score": response.agency_score,
        "output_score": response.output_score,
        "evidence": response.evidence,
        "module_risks": [mr.model_dump() for mr in response.module_risks],
        "latency_ms": response.latency_ms or 0,
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    """Expanded health check: active_guards, wrapper_coverage, chroma, ollama, fusion."""
    try:
        report = get_health_report()
        return {
            "status": report.status,
            "passed": report.passed,
            "failed": report.failed,
            "total": report.total,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail, "latency_ms": c.latency_ms}
                for c in report.checks
            ],
        }
    except Exception as e:
        return {
            "status": "CRITICAL",
            "passed": 0,
            "failed": 1,
            "total": 1,
            "checks": [{"name": "healthcheck", "status": "FAIL", "detail": str(e), "latency_ms": 0}],
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
