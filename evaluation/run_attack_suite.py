"""
evaluation/run_attack_suite.py
==================================
Sistematik saldırı çalıştırıcı.

Pipeline:
    dataset → request builder → gateway (POST /analyze) → result parser → CSV

Tüm attack datasetlerini sırayla çalıştırır, gateway'e gönderir,
sonuçları tek CSV'ye yazar.

Datasets:
    1. injection_dataset_v1.csv          → prompt injection attacks
    2. advanced_poison_samples.json      → RAG poisoning attacks
    3. agency_attack_scenarios.json      → tool misuse attacks

Çıktı:
    runs/fusion_attack_results.csv

Usage:
    # Gateway çalışıyorken:
    python evaluation/run_attack_suite.py

    # Gateway olmadan (doğrudan engine kullanarak):
    python evaluation/run_attack_suite.py --direct
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_FILE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FILE_DIR.parent if _FILE_DIR.name == "evaluation" else _FILE_DIR
_RUNS_DIR = _PROJECT_ROOT / "runs"
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Request builder
# ---------------------------------------------------------------------------
def build_prompt_attacks() -> List[Dict]:
    """Load prompt injection dataset."""
    csv_path = _PROJECT_ROOT / "datasets" / "injection_prompts" / "injection_dataset_v1.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return []

    attacks = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if row.get("label", "0") == "1":  # only attacks
                attacks.append({
                    "attack_id": f"PROMPT-{i+1:03d}",
                    "attack_type": "prompt_injection",
                    "module_targeted": "prompt_guard",
                    "request": {
                        "user_input": row["prompt"],
                        "role": "basic",
                    },
                    "dataset_source": "injection_dataset_v1.csv",
                })
    return attacks


def build_rag_attacks() -> List[Dict]:
    """Load RAG poisoning dataset."""
    json_path = _PROJECT_ROOT / "datasets" / "poisoned_corpus" / "advanced_poison_samples.json"
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    attacks = []
    for doc in dataset["documents"]:
        if doc.get("is_poisoned"):
            attacks.append({
                "attack_id": doc["doc_id"],
                "attack_type": f"rag_{doc.get('poison_technique', 'unknown')}",
                "module_targeted": "rag_guard",
                "request": {
                    "user_input": doc.get("target_query", "What is the policy?"),
                    "retrieved_context": doc["content"],
                    "role": "basic",
                },
                "dataset_source": "advanced_poison_samples.json",
            })
    return attacks


def build_agency_attacks() -> List[Dict]:
    """Load agency attack scenarios."""
    json_path = _PROJECT_ROOT / "datasets" / "output_agency_attacks" / "agency_attack_scenarios.json"
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    attacks = []
    for s in dataset["scenarios"]:
        if s["expected_decision"] == "block":  # only attack scenarios
            attacks.append({
                "attack_id": s["id"],
                "attack_type": f"agency_{s['category']}",
                "module_targeted": "output_agency",
                "request": {
                    "user_input": s.get("description", ""),
                    "tool_call": {"tool": s["tool"], "args": s["args"]},
                    "user_id": s["user_id"],
                    "role": s.get("role", "basic"),
                },
                "dataset_source": "agency_attack_scenarios.json",
            })
    return attacks


# ---------------------------------------------------------------------------
# Gateway caller
# ---------------------------------------------------------------------------
def call_gateway_http(request: Dict, base_url: str = "http://localhost:8000") -> Dict:
    """Send request to gateway via HTTP."""
    import requests as req
    try:
        resp = req.post(f"{base_url}/analyze", json=request, timeout=30)
        return resp.json()
    except Exception as e:
        return {"final_decision": "error", "fused_risk": 0.0, "module_risks": [],
                "latency_ms": 0, "error": str(e)}


def call_gateway_direct(request: Dict) -> Dict:
    """Call engine directly without HTTP."""
    from fusion_gateway.engine import FusionEngine
    engine = FusionEngine()
    response = engine.analyze(
        user_input=request.get("user_input", ""),
        retrieved_context=request.get("retrieved_context"),
        tool_call=request.get("tool_call"),
        role=request.get("role", "basic"),
        user_id=request.get("user_id", "anonymous"),
    )
    return response.to_dict()


# ---------------------------------------------------------------------------
# Result parser
# ---------------------------------------------------------------------------
def parse_result(attack: Dict, response: Dict) -> Dict:
    """Parse gateway response into CSV row."""
    module_risks = response.get("module_risks", [])

    prompt_score = 0.0
    rag_score = 0.0
    agency_score = 0.0

    for m in module_risks:
        if m.get("module") == "prompt_guard":
            prompt_score = m.get("risk_score", 0.0)
        elif m.get("module") == "rag_guard":
            rag_score = m.get("risk_score", 0.0)
        elif m.get("module") == "output_agency":
            agency_score = m.get("risk_score", 0.0)

    return {
        "attack_id": attack["attack_id"],
        "attack_type": attack["attack_type"],
        "module_targeted": attack["module_targeted"],
        "module_prompt_score": round(prompt_score, 4),
        "module_rag_score": round(rag_score, 4),
        "module_agency_score": round(agency_score, 4),
        "fused_risk_score": round(response.get("fused_risk", 0.0), 4),
        "decision": response.get("final_decision", "error"),
        "latency_ms": response.get("latency_ms", 0),
        "dataset_source": attack["dataset_source"],
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_attack_suite(use_http: bool = False, base_url: str = "http://localhost:8000") -> Dict:
    """Run all attack datasets through the gateway."""

    print(f"\n{'='*65}")
    print(f"  ATTACK SUITE RUNNER")
    print(f"  Mode: {'HTTP → ' + base_url if use_http else 'Direct (no HTTP)'}")
    print(f"{'='*65}")

    # Build all attacks
    all_attacks = []

    print(f"\n  Loading datasets...")
    prompt_attacks = build_prompt_attacks()
    print(f"    Prompt injection: {len(prompt_attacks)} attacks")
    all_attacks.extend(prompt_attacks)

    rag_attacks = build_rag_attacks()
    print(f"    RAG poisoning:    {len(rag_attacks)} attacks")
    all_attacks.extend(rag_attacks)

    agency_attacks = build_agency_attacks()
    print(f"    Agency misuse:    {len(agency_attacks)} attacks")
    all_attacks.extend(agency_attacks)

    print(f"    TOTAL:            {len(all_attacks)} attacks")

    # Run attacks
    results = []
    caller = call_gateway_http if use_http else call_gateway_direct

    print(f"\n  Running attacks...")
    for i, attack in enumerate(all_attacks):
        t0 = time.time()
        response = caller(attack["request"])
        elapsed = int((time.time() - t0) * 1000)

        # Override latency if direct mode didn't set it
        if response.get("latency_ms", 0) == 0:
            response["latency_ms"] = elapsed

        row = parse_result(attack, response)
        results.append(row)

        if (i + 1) % 20 == 0 or (i + 1) == len(all_attacks):
            print(f"    Processed {i+1}/{len(all_attacks)}...")

    # Summary
    decisions = {}
    targeted = {}
    for r in results:
        d = r["decision"]
        decisions[d] = decisions.get(d, 0) + 1
        t = r["module_targeted"]
        if t not in targeted:
            targeted[t] = {"total": 0, "blocked": 0, "allowed": 0}
        targeted[t]["total"] += 1
        if d in ("block", "flag"):
            targeted[t]["blocked"] += 1
        else:
            targeted[t]["allowed"] += 1

    print(f"\n{'='*65}")
    print(f"  ATTACK SUITE RESULTS")
    print(f"{'='*65}")
    print(f"  Total attacks: {len(results)}")
    print(f"  Decision distribution: {decisions}")
    print(f"\n  Per-module:")
    for mod, stats in targeted.items():
        block_rate = stats["blocked"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"    {mod:20s}: {stats['blocked']}/{stats['total']} blocked ({block_rate:.1f}%)")

    # Save CSV
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RUNS_DIR / "fusion_attack_results.csv"
    fieldnames = [
        "attack_id", "attack_type", "module_targeted",
        "module_prompt_score", "module_rag_score", "module_agency_score",
        "fused_risk_score", "decision", "latency_ms", "dataset_source",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  [Saved] {csv_path} ({len(results)} rows)")

    summary = {
        "total_attacks": len(results),
        "decisions": decisions,
        "per_module": targeted,
    }
    summary_path = _RUNS_DIR / "fusion_attack_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  [Saved] {summary_path}")

    print(f"\n{'='*65}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attack suite through gateway")
    parser.add_argument("--http", action="store_true", help="Use HTTP (gateway must be running)")
    parser.add_argument("--url", default="http://localhost:8000", help="Gateway URL")
    parser.add_argument("--direct", action="store_true", default=True, help="Direct engine call (default)")
    args = parser.parse_args()

    use_http = args.http
    run_attack_suite(use_http=use_http, base_url=args.url)
