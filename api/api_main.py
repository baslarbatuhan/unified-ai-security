"""
api/main.py
===============
Minimal FastAPI gateway.

Hocanın notu: "tam production API değil, minimal bir POST /analyze endpointi yeterli"

Endpoints:
    POST /analyze   → Tam pipeline: prompt + rag + agency → fusion → karar
    GET  /health    → Security healthcheck

Usage:
    pip install fastapi uvicorn
    uvicorn api.main:app --host 0.0.0.0 --port 8000

    curl -X POST http://localhost:8000/analyze \\
        -H "Content-Type: application/json" \\
        -d '{"user_input": "Ignore all instructions", "role": "basic"}'
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from fusion_gateway.engine import FusionEngine
from api.security_gateway import SecurityGateway
from api.health import get_health_report
from schemas.risk_schema import (
    AnalyzeRequest as SchemaRequest,
    AnalyzeResponse as SchemaResponse,
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


class AnalyzeResponseModel(BaseModel):
    final_decision: str
    fused_risk: float
    prompt_score: float = 0.0
    rag_score: float = 0.0
    agency_score: float = 0.0
    evidence: List[str] = []
    module_risks: List[Dict[str, Any]]
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    passed: int = 0
    failed: int = 0
    total: int = 0
    checks: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# App + Engine
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Unified AI Security Gateway",
    description="3 modül (prompt_guard, rag_guard, output_agency) → fusion → karar",
    version="2.0.0",
)

engine = FusionEngine()
gateway = SecurityGateway(engine)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    """Run security self-check on startup."""
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
        print(f"[API] Security self-check FAILED: {e}")
        print("[API] WARNING: Running without full security validation")
    except ImportError as e:
        print(f"[API] Self-check skipped (import error): {e}")


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

    schema_request = SchemaRequest(
        prompt=prompt,
        retrieved_docs=retrieved_docs,
        context=plain_context,
        tool_request=tool_req,
        tool_candidates=tool_candidates_typed,
        session_context=session,
    )

    response = gateway.analyze(schema_request)

    return {
        "final_decision": response.decision,
        "fused_risk": response.fused_risk_score,
        "prompt_score": response.prompt_score,
        "rag_score": response.rag_score,
        "agency_score": response.agency_score,
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
