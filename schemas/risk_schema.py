"""Pydantic schemas for module risk outputs and gateway IO.

This file is intentionally minimal; expand as the project grows.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

Decision = Literal["allow", "sanitize", "block", "flag"]


class ModuleRisk(BaseModel):
    module: str = Field(..., description="Module identifier, e.g., output_agency | prompt_guard | rag_guard")
    risk_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    decision: Decision
    evidence: List[str] = Field(default_factory=list)
    latency_ms: Optional[int] = Field(default=None, ge=0)


# ---------------------------------------------------------------------------
# Tool request (agency guard input)
# ---------------------------------------------------------------------------
class ToolRequest(BaseModel):
    tool: str = Field(..., description="Tool name, e.g., get_order")
    params: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Session context (user identity + role)
# ---------------------------------------------------------------------------
class SessionContext(BaseModel):
    user_id: str = Field(default="anonymous")
    role: str = Field(default="basic")
    session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Gateway request — expanded
# ---------------------------------------------------------------------------
class ConfigOverrides(BaseModel):
    """Per-request configuration overrides — what the dashboard sliders send.

    Every field is optional; omitted fields fall back to the gateway's
    YAML-loaded defaults. Overrides are applied **only for this request**;
    the long-lived FusionEngine instance is not mutated, so concurrent
    requests with different overrides do not interfere.

    Keys for `weights` / `modules_enabled`: `prompt_guard`, `rag_guard`,
    `output_agency`, `output_guard`. Keys for `thresholds`: `allow`,
    `sanitize`, `block`. Keys for `override`: `critical_threshold`,
    `critical_multiplier`, `elevated_threshold`, `elevated_multiplier`.
    """

    weights: Optional[Dict[str, float]] = None
    thresholds: Optional[Dict[str, float]] = None
    override: Optional[Dict[str, float]] = None
    modules_enabled: Optional[Dict[str, bool]] = None


class AnalyzeRequest(BaseModel):
    prompt: str = Field(..., description="User prompt text")
    retrieved_docs: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="RAG retrieved documents, each with 'doc_id' and 'content'",
    )
    context: Optional[str] = Field(
        default=None,
        description="Plain retrieved text when structured docs are not sent (public alias)",
    )
    tool_request: Optional[ToolRequest] = None
    tool_candidates: Optional[List[ToolRequest]] = Field(
        default=None,
        description="If tool_request is omitted, the first candidate is used by the gateway",
    )
    session_context: SessionContext = Field(default_factory=SessionContext)
    config_overrides: Optional[ConfigOverrides] = Field(
        default=None,
        description="Per-request fusion config — overrides gateway defaults for this call only.",
    )

    # Backward-compat aliases (read-only, not required)
    user_input: Optional[str] = Field(default=None, exclude=True)
    retrieved_context: Optional[str] = Field(default=None, exclude=True)
    role: str = Field(default="basic", exclude=True)

    def get_prompt(self) -> str:
        """Return prompt, falling back to legacy user_input."""
        return self.prompt or self.user_input or ""

    def get_role(self) -> str:
        return self.session_context.role or self.role

    def get_user_id(self) -> str:
        return self.session_context.user_id


# ---------------------------------------------------------------------------
# Gateway response — expanded
# ---------------------------------------------------------------------------
class AnalyzeResponse(BaseModel):
    decision: Decision = Field(..., description="Final gateway decision")
    fused_risk_score: float = Field(..., ge=0.0, le=1.0)
    prompt_score: float = Field(default=0.0, ge=0.0, le=1.0)
    rag_score: float = Field(default=0.0, ge=0.0, le=1.0)
    agency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    # Populated only by /analyze-output (post-LLM screening). The plain
    # /analyze path never sets this — stays at 0.0 for backward compat.
    output_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    module_risks: List[ModuleRisk] = Field(default_factory=list)
    latency_ms: Optional[int] = Field(default=None, ge=0)

    # Backward-compat aliases
    @property
    def final_decision(self) -> Decision:
        return self.decision

    @property
    def fused_risk(self) -> float:
        return self.fused_risk_score
