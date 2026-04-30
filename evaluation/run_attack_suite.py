"""
evaluation/run_attack_suite.py
==================================
Sistematik saldiri calistiricisi (HTTP-only).

Pipeline:
    dataset → request builder → gateway (POST /analyze) → result parser → CSV

Tum attack datasetlerini sirayla calistirir, gateway'e gonderir,
sonuclari tek CSV'ye yazar.

Datasets:
    1. injection_dataset_v1.csv          → prompt injection attacks
    2. advanced_poison_samples.json      → RAG poisoning attacks
    3. agency_attack_scenarios.json      → tool misuse attacks

Cikti:
    runs/gateway_attack_results.csv

Usage:
    # Gateway calisiyor olmali:
    python evaluation/run_attack_suite.py
    python evaluation/run_attack_suite.py --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_FILE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FILE_DIR.parent if _FILE_DIR.name == "evaluation" else _FILE_DIR
_RUNS_DIR = _PROJECT_ROOT / "runs"
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Request builders — use new schema format (AnalyzeRequest)
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
                        "prompt": row["prompt"],
                        "session_context": {"user_id": "attacker", "role": "basic"},
                    },
                    "dataset_source": "injection_dataset_v1.csv",
                })
    return attacks


def build_rag_attacks() -> List[Dict]:
    """Load RAG poisoning dataset.

    Each poisoned doc is sent alongside 3-4 clean docs to simulate
    a realistic top-k RAG retrieval (real-world RAG never returns a
    single document).
    """
    json_path = _PROJECT_ROOT / "datasets" / "poisoned_corpus" / "advanced_poison_samples.json"
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Collect clean docs to use as padding
    clean_docs = [
        {"doc_id": doc["doc_id"], "content": doc["content"]}
        for doc in dataset["documents"]
        if not doc.get("is_poisoned")
    ]

    attacks = []
    for doc in dataset["documents"]:
        if doc.get("is_poisoned"):
            # Build realistic top-k: poisoned doc at random rank among 5 docs
            padding = random.sample(clean_docs, min(4, len(clean_docs)))
            retrieved = [{"doc_id": doc["doc_id"], "content": doc["content"]}]
            retrieved.extend(padding)
            random.shuffle(retrieved)

            attacks.append({
                "attack_id": doc["doc_id"],
                "attack_type": f"rag_{doc.get('poison_technique', 'unknown')}",
                "module_targeted": "rag_guard",
                "request": {
                    "prompt": doc.get("target_query", "What is the policy?"),
                    "retrieved_docs": retrieved,
                    "session_context": {"user_id": "user_test", "role": "basic"},
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
                    "prompt": s.get("description", ""),
                    "tool_request": {"tool": s["tool"], "params": s["args"]},
                    "session_context": {
                        "user_id": s["user_id"],
                        "role": s.get("role", "basic"),
                    },
                },
                "dataset_source": "agency_attack_scenarios.json",
            })
    return attacks


# ---------------------------------------------------------------------------
# Gateway caller (HTTP only) — rate-limit-aware
# ---------------------------------------------------------------------------
def call_gateway(
    request: Dict,
    base_url: str = "http://localhost:8000",
    *,
    max_retries: int = 3,
) -> Dict:
    """Send POST /analyze to the gateway, retrying on 429 with backoff.

    The gateway's `RateLimitMiddleware` (configs/service_limits.yaml,
    default 60 RPM / 10 burst) returns 429 with a `Retry-After` header
    when a token-bucket runs dry. Earlier the runner mapped 429 onto
    `decision="error"` and moved on — so a single warmed-up attack
    suite (147 cases) appeared to "fail" 130/147 cases and corrupted
    the gateway_attack_results.csv → baseline_comparison pipeline.

    We now respect Retry-After (capped at 30 s) and retry up to
    `max_retries` times before falling back to `error`.
    """
    import requests as req
    delay = 1.0
    for attempt in range(max_retries + 1):
        try:
            resp = req.post(f"{base_url}/analyze", json=request, timeout=60)
            if resp.status_code == 429 and attempt < max_retries:
                ra = resp.headers.get("Retry-After")
                wait = min(float(ra), 30.0) if ra else delay
                time.sleep(wait)
                delay = min(delay * 2, 30.0)
                continue
            resp.raise_for_status()
            return resp.json()
        except req.exceptions.HTTPError as e:
            return {
                "decision": "error",
                "fused_risk_score": 0.0,
                "prompt_score": 0.0,
                "rag_score": 0.0,
                "agency_score": 0.0,
                "evidence": [f"HTTP {resp.status_code}: {resp.text[:200]}"],
                "module_risks": [],
                "latency_ms": 0,
                "error": str(e),
            }
        except Exception as e:
            return {
                "decision": "error",
                "fused_risk_score": 0.0,
                "prompt_score": 0.0,
                "rag_score": 0.0,
                "agency_score": 0.0,
                "evidence": [str(e)],
                "module_risks": [],
                "latency_ms": 0,
                "error": str(e),
            }
    return {
        "decision": "error",
        "fused_risk_score": 0.0,
        "prompt_score": 0.0,
        "rag_score": 0.0,
        "agency_score": 0.0,
        "evidence": [f"exceeded {max_retries} retries on 429"],
        "module_risks": [],
        "latency_ms": 0,
        "error": "rate_limited",
    }


# ---------------------------------------------------------------------------
# Result parser — new schema format
# ---------------------------------------------------------------------------
def parse_result(attack: Dict, response: Dict) -> Dict:
    """Parse gateway response into CSV row."""
    # New schema: direct top-level scores
    prompt_score = response.get("prompt_score", 0.0)
    rag_score = response.get("rag_score", 0.0)
    agency_score = response.get("agency_score", 0.0)

    # Fallback: extract from module_risks if top-level not present
    if prompt_score == 0.0 and rag_score == 0.0 and agency_score == 0.0:
        for m in response.get("module_risks", []):
            if m.get("module") == "prompt_guard":
                prompt_score = m.get("risk_score", 0.0)
            elif m.get("module") == "rag_guard":
                rag_score = m.get("risk_score", 0.0)
            elif m.get("module") == "output_agency":
                agency_score = m.get("risk_score", 0.0)

    # Determine override_triggered: check if max-rule override was active
    fused = response.get("fused_risk_score", response.get("fused_risk", 0.0))
    module_max = max(prompt_score, rag_score, agency_score)
    override_triggered = "yes" if module_max >= 0.60 and fused > (
        0.30 * prompt_score + 0.30 * rag_score + 0.40 * agency_score + 0.01
    ) else "no"

    # Extract per-module latency from module_risks
    prompt_latency = 0
    rag_latency = 0
    agency_latency = 0
    for m in response.get("module_risks", []):
        mod = m.get("module", "")
        lat = m.get("latency_ms", 0) or 0
        if mod == "prompt_guard":
            prompt_latency = lat
        elif mod == "rag_guard":
            rag_latency = lat
        elif mod == "output_agency":
            agency_latency = lat

    total_latency = response.get("latency_ms", 0)
    fusion_latency = max(0, total_latency - max(prompt_latency, rag_latency, agency_latency))

    return {
        "attack_id": attack["attack_id"],
        "attack_type": attack["attack_type"],
        "module_targeted": attack["module_targeted"],
        "module_prompt_score": round(prompt_score, 4),
        "module_rag_score": round(rag_score, 4),
        "module_agency_score": round(agency_score, 4),
        "fused_risk_score": round(fused, 4),
        "override_triggered": override_triggered,
        "decision": response.get("decision", response.get("final_decision", "error")),
        "latency_ms": total_latency,
        "prompt_latency_ms": prompt_latency,
        "rag_latency_ms": rag_latency,
        "agency_latency_ms": agency_latency,
        "fusion_latency_ms": fusion_latency,
        "dataset_source": attack["dataset_source"],
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_attack_suite(
    base_url: str = "http://localhost:8000",
    seed: int = 42,
    rps: float = 1.0,
) -> Dict:
    """Run all attack datasets through the HTTP gateway.

    `rps` (requests per second) paces the runner so it doesn't trip the
    gateway's RateLimitMiddleware (default 60 RPM / 10 burst). 1.0 RPS
    matches the default tier exactly; lower for slower hardware, raise
    only if you've widened service_limits.yaml.
    """
    # Set seed for reproducibility
    random.seed(seed)
    min_interval = 1.0 / rps if rps > 0 else 0.0

    print(f"\n{'='*65}")
    print(f"  ATTACK SUITE RUNNER (HTTP)")
    print(f"  Gateway: {base_url}")
    print(f"  Seed: {seed}")
    print(f"  Pacing: {rps} req/s ({min_interval*1000:.0f} ms gap)")
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

    if not all_attacks:
        print("\n  ERROR: No attacks loaded. Check dataset paths.")
        return {"error": "no_attacks"}

    # Run attacks
    results = []

    print(f"\n  Running attacks...")
    last_call_at = 0.0
    for i, attack in enumerate(all_attacks):
        # Throttle to stay under the gateway's rate limit. Sleep is
        # measured from the *start* of the previous request so genuinely
        # slow gateway responses naturally absorb the gap.
        gap = (last_call_at + min_interval) - time.time()
        if gap > 0:
            time.sleep(gap)
        last_call_at = time.time()

        t0 = time.time()
        response = call_gateway(attack["request"], base_url)
        elapsed = int((time.time() - t0) * 1000)

        # Override latency if gateway didn't set it
        if response.get("latency_ms", 0) == 0:
            response["latency_ms"] = elapsed

        row = parse_result(attack, response)
        results.append(row)

        if (i + 1) % 20 == 0 or (i + 1) == len(all_attacks):
            print(f"    Processed {i+1}/{len(all_attacks)}...")

    # Summary
    decisions = {}
    targeted = {}
    override_count = 0
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
        if r["override_triggered"] == "yes":
            override_count += 1

    print(f"\n{'='*65}")
    print(f"  ATTACK SUITE RESULTS")
    print(f"{'='*65}")
    print(f"  Total attacks: {len(results)}")
    print(f"  Decision distribution: {decisions}")
    print(f"  Override triggered: {override_count}/{len(results)}")
    print(f"\n  Per-module:")
    for mod, stats in targeted.items():
        block_rate = stats["blocked"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"    {mod:20s}: {stats['blocked']}/{stats['total']} blocked ({block_rate:.1f}%)")

    # Save CSV — new filename: gateway_attack_results.csv
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _RUNS_DIR / "gateway_attack_results.csv"
    fieldnames = [
        "attack_id", "attack_type", "module_targeted",
        "module_prompt_score", "module_rag_score", "module_agency_score",
        "fused_risk_score", "override_triggered", "decision",
        "latency_ms", "prompt_latency_ms", "rag_latency_ms",
        "agency_latency_ms", "fusion_latency_ms", "dataset_source",
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
        "override_triggered": override_count,
    }
    summary_path = _RUNS_DIR / "gateway_attack_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  [Saved] {summary_path}")

    print(f"\n{'='*65}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run attack suite through HTTP gateway")
    parser.add_argument("--url", default="http://localhost:8000", help="Gateway URL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--rps", type=float, default=1.0,
        help=(
            "Pacing: requests per second sent to the gateway. Default 1.0 "
            "matches the default RateLimitMiddleware tier (60 RPM). Lower "
            "for slow hardware; raise only after widening service_limits.yaml."
        ),
    )
    args = parser.parse_args()

    run_attack_suite(base_url=args.url, seed=args.seed, rps=args.rps)
