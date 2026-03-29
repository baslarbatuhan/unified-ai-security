"""Pydantic schemas for module risk outputs and gateway IO.

This file is intentionally minimal; expand as the project grows.
"""
from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field

Decision = Literal["allow", "sanitize", "block", "flag"]

class ModuleRisk(BaseModel):
    module: str = Field(..., description="Module identifier, e.g., output_agency | prompt_guard | rag_guard")
    risk_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    decision: Decision
    evidence: List[str] = Field(default_factory=list)
    latency_ms: Optional[int] = Field(default=None, ge=0)

class AnalyzeRequest(BaseModel):
    user_input: str
    retrieved_context: Optional[str] = None
    role: str = "basic"

class AnalyzeResponse(BaseModel):
    final_decision: Decision
    fused_risk: float = Field(..., ge=0.0, le=1.0)
    module_risks: List[ModuleRisk]
