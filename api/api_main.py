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


# ---------------------------------------------------------------------------
# Pydantic models (compatible with schemas/risk_schema.py)
# ---------------------------------------------------------------------------
class AnalyzeRequestModel(BaseModel):
    user_input: str = ""
    retrieved_context: Optional[str] = None
    tool_call: Optional[Dict[str, Any]] = None
    role: str = "basic"
    user_id: str = "anonymous"


class ModuleRiskModel(BaseModel):
    module: str
    risk_score: float
    confidence: float
    decision: str
    evidence: List[str] = []
    latency_ms: Optional[int] = None


class AnalyzeResponseModel(BaseModel):
    final_decision: str
    fused_risk: float
    module_risks: List[Dict[str, Any]]
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    checks: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# App + Engine
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Unified AI Security Gateway",
    description="3 modül (prompt_guard, rag_guard, output_agency) → fusion → karar",
    version="1.0.0",
)

engine = FusionEngine()


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

    Request body:
        user_input:        Kullanıcı promptu
        retrieved_context: RAG retrieval sonucu (opsiyonel)
        tool_call:         Tool çağrısı {"tool": "...", "args": {...}} (opsiyonel)
        role:              Kullanıcı rolü (default: "basic")
        user_id:           Kullanıcı ID (default: "anonymous")

    Response:
        final_decision:  allow / sanitize / flag / block
        fused_risk:      0.0 - 1.0
        module_risks:    3 modülün ayrı ayrı sonuçları
        latency_ms:      Toplam işlem süresi
    """
    response = engine.analyze(
        user_input=request.user_input,
        retrieved_context=request.retrieved_context,
        tool_call=request.tool_call,
        role=request.role,
        user_id=request.user_id,
    )
    return response.to_dict()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    """Sistem sağlığı kontrolü."""
    try:
        from evaluation.security_healthcheck import run_healthcheck
        result = run_healthcheck()
        return {
            "status": result.overall_status,
            "checks": result.checks,
        }
    except Exception as e:
        return {
            "status": "DEGRADED",
            "checks": [{"name": "healthcheck", "status": "FAIL", "detail": str(e)}],
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
