"""
evaluation/generate_attack_matrix.py
========================================
Saldırı matrisi oluşturucu.

Tüm test sonuçlarını okuyarak birleşik saldırı matrisi oluşturur.

Matrix kolonları:
    attack_id, attack_type, module_targeted, input_artifact,
    expected_defense, observed_decision, risk_score,
    success_or_failure, dataset_source, notes

Çıktı:
    reports/attack_matrix.csv

Usage:
    python evaluation/generate_attack_matrix.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

_FILE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FILE_DIR.parent if _FILE_DIR.name == "evaluation" else _FILE_DIR
_RUNS_DIR = _PROJECT_ROOT / "runs"
_REPORTS_DIR = _PROJECT_ROOT / "reports"
sys.path.insert(0, str(_PROJECT_ROOT))


def load_prompt_results() -> List[Dict]:
    """Load prompt injection + evasion results."""
    rows = []

    # Hafta 2 combined results
    csv_path = _RUNS_DIR / "prompt_metrics.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, r in enumerate(reader):
                if r.get("label", r.get("is_attack", "0")) == "1":
                    detected = r.get("combined_detected", r.get("detected", "False"))
                    decision = "block" if detected == "True" else "allow"
                    rows.append({
                        "attack_id": f"PI-{i+1:03d}",
                        "attack_type": "prompt_injection",
                        "module_targeted": "prompt_guard",
                        "input_artifact": (r.get("prompt", "")[:80] + "...") if len(r.get("prompt", "")) > 80 else r.get("prompt", ""),
                        "expected_defense": "block",
                        "observed_decision": decision,
                        "risk_score": r.get("risk_score", r.get("combined_score", "0")),
                        "success_or_failure": "success" if decision in ("block", "flag", "sanitize") else "failure",
                        "dataset_source": "injection_dataset_v1.csv",
                        "notes": f"semantic+pattern combined",
                    })

    # Hafta 3 evasion results
    csv_path = _RUNS_DIR / "prompt_evasion_metrics.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, r in enumerate(reader):
                detected = r.get("detected", "False") == "True"
                rows.append({
                    "attack_id": f"PE-{i+1:03d}",
                    "attack_type": f"evasion_{r.get('attack_type', 'unknown')}",
                    "module_targeted": "prompt_guard",
                    "input_artifact": (r.get("prompt_variant", "")[:80] + "...") if len(r.get("prompt_variant", "")) > 80 else r.get("prompt_variant", ""),
                    "expected_defense": "block",
                    "observed_decision": r.get("decision", "allow"),
                    "risk_score": r.get("risk_score", "0"),
                    "success_or_failure": "success" if detected else "failure",
                    "dataset_source": "evasion_variants",
                    "notes": f"raw={r.get('raw_semantic_score','')}, norm={r.get('normalized_semantic_score','')}",
                })

    return rows


def load_rag_results() -> List[Dict]:
    """Load RAG poisoning results."""
    rows = []

    # Hafta 2 original
    csv_path = _RUNS_DIR / "rag_metrics.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("is_poisoned", "False") == "True":
                    detected = r.get("detected", "False") == "True"
                    rows.append({
                        "attack_id": r.get("test_case", r.get("doc_id", "")),
                        "attack_type": f"rag_{r.get('poison_type', 'unknown')}",
                        "module_targeted": "rag_guard",
                        "input_artifact": r.get("test_case", "")[:80],
                        "expected_defense": "sanitize",
                        "observed_decision": r.get("decision", "allow"),
                        "risk_score": r.get("risk_score", "0"),
                        "success_or_failure": "success" if detected else "failure",
                        "dataset_source": "poison_samples.json",
                        "notes": f"poison_score={r.get('poison_score', '')}",
                    })

    # Hafta 3 advanced
    csv_path = _RUNS_DIR / "rag_poison_metrics.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("is_poisoned", "False") == "True":
                    detected = r.get("detected", "False") == "True"
                    rows.append({
                        "attack_id": r.get("test_case", ""),
                        "attack_type": f"rag_adv_{r.get('poison_technique', 'unknown')}",
                        "module_targeted": "rag_guard",
                        "input_artifact": r.get("test_case", "")[:80],
                        "expected_defense": "sanitize",
                        "observed_decision": r.get("decision", "allow"),
                        "risk_score": r.get("risk_score", "0"),
                        "success_or_failure": "success" if detected else "failure",
                        "dataset_source": "advanced_poison_samples.json",
                        "notes": f"technique={r.get('poison_technique','')}, score={r.get('poison_score','')}",
                    })

    return rows


def load_agency_results() -> List[Dict]:
    """Load agency attack results."""
    rows = []

    # Hafta 2 hardcoded
    csv_path = _RUNS_DIR / "agency_metrics.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("expected", "") == "block":
                    correct = r.get("correct", "False") == "True"
                    rows.append({
                        "attack_id": f"AG-HC-{r.get('test_case', '')[:20]}",
                        "attack_type": f"agency_{r.get('test_category', 'unknown')}",
                        "module_targeted": "output_agency",
                        "input_artifact": r.get("test_case", "")[:80],
                        "expected_defense": "block",
                        "observed_decision": r.get("decision", "allow"),
                        "risk_score": r.get("risk_score", "0"),
                        "success_or_failure": "success" if correct else "failure",
                        "dataset_source": "test_id_enumeration.py (hardcoded)",
                        "notes": f"category={r.get('test_category', '')}",
                    })

    # Hafta 3 dataset-driven
    csv_path = _RUNS_DIR / "agency_attack_metrics.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("expected_decision", "") == "block":
                    correct = r.get("correct", "False") == "True"
                    rows.append({
                        "attack_id": r.get("test_case", ""),
                        "attack_type": f"agency_{r.get('category', 'unknown')}",
                        "module_targeted": "output_agency",
                        "input_artifact": r.get("description", "")[:80],
                        "expected_defense": "block",
                        "observed_decision": r.get("decision", "allow"),
                        "risk_score": r.get("risk_score", "0"),
                        "success_or_failure": "success" if correct else "failure",
                        "dataset_source": "agency_attack_scenarios.json",
                        "notes": f"category={r.get('category','')}, reason={r.get('block_reason','')}",
                    })

    return rows


def generate_attack_matrix() -> Dict:
    """Generate the full attack matrix."""

    print(f"\n{'='*65}")
    print(f"  ATTACK MATRIX GENERATOR")
    print(f"{'='*65}")

    all_rows = []

    prompt_rows = load_prompt_results()
    print(f"  Prompt attacks: {len(prompt_rows)}")
    all_rows.extend(prompt_rows)

    rag_rows = load_rag_results()
    print(f"  RAG attacks:    {len(rag_rows)}")
    all_rows.extend(rag_rows)

    agency_rows = load_agency_results()
    print(f"  Agency attacks: {len(agency_rows)}")
    all_rows.extend(agency_rows)

    print(f"  TOTAL:          {len(all_rows)} rows")

    # Summary
    success = sum(1 for r in all_rows if r["success_or_failure"] == "success")
    failure = sum(1 for r in all_rows if r["success_or_failure"] == "failure")
    total = len(all_rows)
    defense_rate = success / total * 100 if total > 0 else 0

    # Per-module
    modules = {}
    for r in all_rows:
        m = r["module_targeted"]
        if m not in modules:
            modules[m] = {"total": 0, "success": 0, "failure": 0}
        modules[m]["total"] += 1
        modules[m][r["success_or_failure"]] += 1

    print(f"\n  DEFENSE SUMMARY:")
    print(f"  Total attacks: {total} | Defended: {success} | Evaded: {failure} | Rate: {defense_rate:.1f}%")
    print(f"\n  Per-module:")
    for mod, stats in sorted(modules.items()):
        rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"    {mod:20s}: {stats['success']}/{stats['total']} defended ({rate:.1f}%)")

    # Per-attack-type
    attack_types = {}
    for r in all_rows:
        at = r["attack_type"]
        if at not in attack_types:
            attack_types[at] = {"total": 0, "success": 0}
        attack_types[at]["total"] += 1
        if r["success_or_failure"] == "success":
            attack_types[at]["success"] += 1

    print(f"\n  Per-attack-type:")
    for at, stats in sorted(attack_types.items()):
        rate = stats["success"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"    {at:35s}: {stats['success']}/{stats['total']} ({rate:.1f}%)")

    # Save CSV
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _REPORTS_DIR / "attack_matrix.csv"
    fieldnames = [
        "attack_id", "attack_type", "module_targeted", "input_artifact",
        "expected_defense", "observed_decision", "risk_score",
        "success_or_failure", "dataset_source", "notes",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  [Saved] {csv_path} ({len(all_rows)} rows)")

    summary = {
        "total_attacks": total,
        "defended": success,
        "evaded": failure,
        "defense_rate": round(defense_rate, 2),
        "per_module": modules,
        "per_attack_type": attack_types,
    }
    summary_path = _REPORTS_DIR / "attack_matrix_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  [Saved] {summary_path}")

    print(f"\n{'='*65}")
    return summary


if __name__ == "__main__":
    generate_attack_matrix()
