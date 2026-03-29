"""
evaluation/run_experiments.py
================================
Unified experiment runner.

Purpose:
    Run ALL security tests with a single command:
    1. Agency tests (ID enumeration, IDOR, anti-enum)
    2. Prompt injection tests (semantic + pattern detection)
    3. RAG poison tests (poison detection in retrieval)

    Collects results and produces a unified summary.

Usage:
    python evaluation/run_experiments.py              # Run all
    python evaluation/run_experiments.py --only agency # Run only agency
    python evaluation/run_experiments.py --only prompt # Run only prompt
    python evaluation/run_experiments.py --only rag    # Run only rag
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

_FILE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FILE_DIR.parent if _FILE_DIR.name == "evaluation" else _FILE_DIR
_RUNS_DIR = _PROJECT_ROOT / "runs"

sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Test runners (imported lazily to handle missing dependencies)
# ---------------------------------------------------------------------------
def run_agency_tests() -> Dict:
    """Run agency / ID enumeration tests."""
    try:
        sys.path.insert(0, str(_PROJECT_ROOT / "tests"))
        from test_id_enumeration import run_id_enumeration_tests
        return {"status": "completed", "results": run_id_enumeration_tests()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def run_prompt_tests() -> Dict:
    """Run prompt injection detection tests."""
    try:
        from evaluation.prompt_injection_tests import run_prompt_injection_tests
        return {"status": "completed", "results": run_prompt_injection_tests()}
    except ImportError:
        try:
            from prompt_injection_tests import run_prompt_injection_tests
            return {"status": "completed", "results": run_prompt_injection_tests()}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def run_rag_tests() -> Dict:
    """Run RAG poison detection tests."""
    try:
        sys.path.insert(0, str(_PROJECT_ROOT / "tests"))
        from test_rag_poison_detection import run_rag_poison_tests
        return {"status": "completed", "results": run_rag_poison_tests()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_all_experiments(only: Optional[str] = None) -> Dict:
    """
    Run all or selected experiments.

    Args:
        only: If set, run only this module ("agency", "prompt", "rag")

    Returns:
        Dict with per-module results and unified summary.
    """
    experiments = {
        "agency": {
            "name": "Excessive Agency / Tool Security",
            "runner": run_agency_tests,
            "enabled": only is None or only == "agency",
        },
        "prompt": {
            "name": "Prompt Injection Detection",
            "runner": run_prompt_tests,
            "enabled": only is None or only == "prompt",
        },
        "rag": {
            "name": "RAG Poison Detection",
            "runner": run_rag_tests,
            "enabled": only is None or only == "rag",
        },
    }

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modules": {},
        "summary": {
            "total_modules": 0,
            "completed": 0,
            "errors": 0,
            "skipped": 0,
        },
    }

    total_time = 0

    for key, exp in experiments.items():
        if not exp["enabled"]:
            report["modules"][key] = {"status": "skipped"}
            report["summary"]["skipped"] += 1
            continue

        report["summary"]["total_modules"] += 1

        print(f"\n{'#'*65}")
        print(f"  RUNNING: {exp['name']}")
        print(f"{'#'*65}")

        t0 = time.time()
        result = exp["runner"]()
        elapsed = round(time.time() - t0, 2)
        total_time += elapsed

        result["elapsed_seconds"] = elapsed
        report["modules"][key] = result

        if result["status"] == "completed":
            report["summary"]["completed"] += 1
            print(f"\n  ✓ {exp['name']} completed in {elapsed}s")
        else:
            report["summary"]["errors"] += 1
            print(f"\n  ✗ {exp['name']} FAILED: {result.get('error', 'unknown')}")

    report["summary"]["total_elapsed_seconds"] = round(total_time, 2)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run all security experiments")
    parser.add_argument("--only", type=str, choices=["agency", "prompt", "rag"],
                        default=None, help="Run only a specific module")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  UNIFIED AI SECURITY — EXPERIMENT RUNNER")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*65}")

    report = run_all_experiments(only=args.only)

    # Summary
    s = report["summary"]
    print(f"\n{'='*65}")
    print(f"  EXPERIMENT SUMMARY")
    print(f"{'='*65}")
    print(f"  Completed:  {s['completed']}/{s['total_modules']}")
    print(f"  Errors:     {s['errors']}")
    print(f"  Skipped:    {s['skipped']}")
    print(f"  Total time: {s['total_elapsed_seconds']}s")

    # Per-module key metrics
    for key, mod in report["modules"].items():
        if mod["status"] == "completed" and "results" in mod:
            results = mod["results"]
            print(f"\n  [{key.upper()}]")
            if key == "agency":
                print(f"    Accuracy: {results.get('accuracy', 'N/A')}")
                print(f"    Block rate: {results.get('unauthorized_access_block_rate', 'N/A')}")
                print(f"    Enum detection: {results.get('enumeration_detection_rate', 'N/A')}")
            elif key == "prompt":
                combined = results.get("combined", {})
                print(f"    Combined F1: {combined.get('f1', 'N/A')}")
                print(f"    Combined Precision: {combined.get('precision', 'N/A')}")
                print(f"    Combined Recall: {combined.get('recall', 'N/A')}")
            elif key == "rag":
                print(f"    F1: {results.get('f1', 'N/A')}")
                print(f"    Precision: {results.get('precision', 'N/A')}")
                print(f"    Recall: {results.get('recall', 'N/A')}")

    print(f"\n{'='*65}")

    # Save unified report
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _RUNS_DIR / "experiment_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  [Saved] {report_path}")

    # Per-module CSVs already saved by individual runners:
    print(f"\n  Individual results:")
    for csv_name in ["agency_metrics.csv", "prompt_metrics.csv", "rag_metrics.csv"]:
        csv_path = _RUNS_DIR / csv_name
        status = "exists" if csv_path.exists() else "not found"
        print(f"    {csv_path}: {status}")


if __name__ == "__main__":
    main()
