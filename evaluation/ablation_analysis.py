"""
evaluation/ablation_analysis.py
===================================
Component Ablation Analysis.

Tests 3 variants to measure each module's contribution:
    1. No LLM judge (embedding only for RAG)
    2. No deobfuscator (raw prompt to semantic eval)
    3. No behavior model (only authz + enum for agency)

Each variant runs the attack suite through a shared engine with the
specified component disabled, then compares detection rates.

Optimized: singleton models loaded once, reused across all variants.

Output:
    runs/analysis_results.csv

Usage:
    python evaluation/ablation_analysis.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

_RUNS_DIR = _PROJECT_ROOT / "runs"


# Shared engine — loaded once, reused across all variants
_shared_engine = None


def _get_engine():
    global _shared_engine
    if _shared_engine is None:
        from fusion_gateway.engine import FusionEngine
        _shared_engine = FusionEngine(parallel=False)
    return _shared_engine


def run_full_pipeline(attacks: List[Dict]) -> Dict:
    """Run attacks through the full pipeline (baseline)."""
    return _run_with_engine(_get_engine(), attacks, "full_pipeline")


def run_no_llm_judge(attacks: List[Dict]) -> Dict:
    """Run attacks with LLM judge disabled (embedding only for RAG)."""
    from fusion_gateway.engine import _get_rag_pipeline
    engine = _get_engine()
    rag_pipeline = _get_rag_pipeline()

    original_ew = rag_pipeline.embedding_weight
    original_jw = rag_pipeline.judge_weight
    rag_pipeline.embedding_weight = 1.0
    rag_pipeline.judge_weight = 0.0

    result = _run_with_engine(engine, attacks, "no_llm_judge")

    rag_pipeline.embedding_weight = original_ew
    rag_pipeline.judge_weight = original_jw
    return result


def run_no_deobfuscator(attacks: List[Dict]) -> Dict:
    """Run attacks with deobfuscator disabled."""
    engine = _get_engine()

    import prompt_guard.deobfuscator as deob_mod
    original_deobfuscate = deob_mod.deobfuscate
    original_report = deob_mod.get_deobfuscation_report
    noop_report = lambda text: {"original": text, "deobfuscated": text, "changed": False, "changes": []}
    deob_mod.deobfuscate = lambda text: text
    deob_mod.get_deobfuscation_report = noop_report

    try:
        import prompt_guard.pipeline as pipe_mod
        pipe_mod.deobfuscate = lambda text: text
        pipe_mod.get_deobfuscation_report = noop_report
    except Exception:
        pipe_mod = None

    result = _run_with_engine(engine, attacks, "no_deobfuscator")

    deob_mod.deobfuscate = original_deobfuscate
    deob_mod.get_deobfuscation_report = original_report
    if pipe_mod:
        pipe_mod.deobfuscate = original_deobfuscate
        pipe_mod.get_deobfuscation_report = original_report

    return result


def run_no_behavior_model(attacks: List[Dict]) -> Dict:
    """Run attacks with behavior model disabled (authz + enum only)."""
    engine = _get_engine()

    import output_agency_defense.behavior_monitor as bm_mod
    original_record = bm_mod.BehaviorMonitor.record

    def noop_record(self, *args, **kwargs):
        from output_agency_defense.behavior_monitor import BehaviorAssessment
        return BehaviorAssessment(risk_level="low", signals=[], risk_score=0.0)

    bm_mod.BehaviorMonitor.record = noop_record
    result = _run_with_engine(engine, attacks, "no_behavior_model")
    bm_mod.BehaviorMonitor.record = original_record
    return result


def _run_with_engine(engine, attacks: List[Dict], variant_name: str) -> Dict:
    """Run all attacks through a configured engine."""
    blocked = 0
    allowed = 0
    flagged = 0

    for i, attack in enumerate(attacks):
        req = attack["request"]
        prompt = req.get("prompt", "")
        retrieved_docs = req.get("retrieved_docs")
        tool_request = req.get("tool_request")
        session = req.get("session_context", {})

        tool_call = None
        if tool_request:
            tool_call = {"tool": tool_request.get("tool", ""), "args": tool_request.get("params", {})}

        response = engine.analyze(
            user_input=prompt,
            retrieved_docs=retrieved_docs,
            tool_call=tool_call,
            role=session.get("role", "basic"),
            user_id=session.get("user_id", "anonymous"),
        )

        if response.final_decision == "block":
            blocked += 1
        elif response.final_decision in ("flag", "sanitize"):
            flagged += 1
        else:
            allowed += 1

        if (i + 1) % 30 == 0:
            print(f"      {i+1}/{len(attacks)}...")

    total = len(attacks)
    return {
        "variant": variant_name,
        "total_attacks": total,
        "blocked": blocked,
        "flagged": flagged,
        "allowed": allowed,
        "detection_rate": round((blocked + flagged) / max(1, total) * 100, 2),
        "block_rate": round(blocked / max(1, total) * 100, 2),
    }


def build_all_attacks() -> List[Dict]:
    """Build all attacks from datasets (same as run_attack_suite)."""
    import random
    random.seed(42)
    from evaluation.run_attack_suite import build_prompt_attacks, build_rag_attacks, build_agency_attacks
    all_attacks = []
    all_attacks.extend(build_prompt_attacks())
    all_attacks.extend(build_rag_attacks())
    all_attacks.extend(build_agency_attacks())
    return all_attacks


def run_ablation():
    """Run full ablation analysis."""
    print(f"\n{'='*65}")
    print(f"  ABLATION ANALYSIS")
    print(f"{'='*65}")

    attacks = build_all_attacks()
    print(f"  Loaded {len(attacks)} attacks\n")

    # Run no-judge first (fastest), then no-deob, no-behavior, then full pipeline last
    # This way the expensive LLM judge calls only happen once (for baseline)
    variants = [
        ("No LLM judge", run_no_llm_judge),
        ("No deobfuscator", run_no_deobfuscator),
        ("No behavior model", run_no_behavior_model),
        ("Full pipeline (baseline)", run_full_pipeline),
    ]

    csv_rows = []
    for name, func in variants:
        print(f"  Running: {name}...")
        t0 = time.time()
        result = func(attacks)
        elapsed = int((time.time() - t0) * 1000)
        result["elapsed_ms"] = elapsed
        csv_rows.append(result)
        print(f"    Detection: {result['detection_rate']}% | Block: {result['block_rate']}% | Time: {elapsed}ms")

    # Reorder: baseline first in CSV
    baseline = next(r for r in csv_rows if r["variant"] == "full_pipeline")
    others = [r for r in csv_rows if r["variant"] != "full_pipeline"]
    csv_rows = [baseline] + others

    # Save CSV
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RUNS_DIR / "analysis_results.csv"
    fieldnames = list(csv_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n  [Saved] {csv_path}")

    # Compare with baseline
    print(f"\n  Impact Analysis (vs baseline {baseline['detection_rate']}%):")
    for row in others:
        diff = row["detection_rate"] - baseline["detection_rate"]
        print(f"    {row['variant']:25s}: {diff:+.2f}% detection change")

    print(f"\n{'='*65}")

    return csv_rows


if __name__ == "__main__":
    run_ablation()
