"""
api/startup.py
=================
Model warmup on application startup.

Preloads all ML models (embedding, LLM judge) with dummy requests
so the first real request is fast.

Usage:
    Called automatically from api_main.py startup event.

Output:
    runs/warmup_latency_metrics.csv
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
_RUNS_DIR = _PROJECT_ROOT / "runs"


def warmup() -> dict:
    """Preload all models with dummy inputs and measure latency.

    Returns:
        dict with per-module warmup latencies in ms.
    """
    results = {}

    # 1. Prompt Guard — loads BGE-M3 embedding model
    print("[Warmup] Loading prompt_guard pipeline...")
    t0 = time.time()
    try:
        from prompt_guard.pipeline import PromptGuardPipeline
        pipeline = PromptGuardPipeline()
        pipeline.run("warmup test prompt")
        results["prompt_guard_ms"] = int((time.time() - t0) * 1000)
        print(f"[Warmup]   prompt_guard OK ({results['prompt_guard_ms']}ms)")
    except Exception as e:
        results["prompt_guard_ms"] = int((time.time() - t0) * 1000)
        print(f"[Warmup]   prompt_guard FAILED: {e}")

    # 2. RAG Guard — loads embedding model + Ollama LLM judge
    print("[Warmup] Loading rag_guard pipeline...")
    t0 = time.time()
    try:
        from fusion_gateway.engine import _get_rag_pipeline
        rag_pipeline = _get_rag_pipeline()
        dummy_docs = [{"doc_id": "warmup_doc", "content": "This is a warmup document for testing."}]
        rag_pipeline.run(dummy_docs, user_query="warmup query")
        results["rag_guard_ms"] = int((time.time() - t0) * 1000)
        print(f"[Warmup]   rag_guard OK ({results['rag_guard_ms']}ms)")
    except Exception as e:
        results["rag_guard_ms"] = int((time.time() - t0) * 1000)
        print(f"[Warmup]   rag_guard FAILED: {e}")

    # 3. Agency Guard — lightweight, no ML models
    print("[Warmup] Loading agency guard...")
    t0 = time.time()
    try:
        from output_agency_defense.resource_registry import create_demo_registry
        from output_agency_defense.object_authz_guard import ObjectAuthzGuard
        from output_agency_defense.anti_enum_guard import AntiEnumGuard
        from output_agency_defense.parameter_validation import ParameterValidator
        create_demo_registry()
        ObjectAuthzGuard(create_demo_registry())
        AntiEnumGuard()
        ParameterValidator()
        results["agency_guard_ms"] = int((time.time() - t0) * 1000)
        print(f"[Warmup]   agency_guard OK ({results['agency_guard_ms']}ms)")
    except Exception as e:
        results["agency_guard_ms"] = int((time.time() - t0) * 1000)
        print(f"[Warmup]   agency_guard FAILED: {e}")

    total = sum(results.values())
    results["total_ms"] = total
    print(f"[Warmup] Models warmed up in {total}ms")

    # Save metrics
    _save_warmup_metrics(results)

    return results


def _save_warmup_metrics(results: dict) -> None:
    """Write warmup latency metrics to CSV."""
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RUNS_DIR / "warmup_latency_metrics.csv"
    fieldnames = ["prompt_guard_ms", "rag_guard_ms", "agency_guard_ms", "total_ms"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: results.get(k, 0) for k in fieldnames})
    print(f"[Warmup] Metrics saved to {csv_path}")
