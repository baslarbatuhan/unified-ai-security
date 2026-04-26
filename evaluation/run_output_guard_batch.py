"""evaluation/run_output_guard_batch.py
Batch-evaluate the output_guard analyzer against a curated dataset and
emit two CSVs (eval-only paths — distinct from the production live
writer in `output_guard/metrics_writer.py` to prevent schema-drift
overwrites):

    runs/output_eval_metrics.csv   — one row per sample with the
                                     decision verdict + per-flag scores
                                     (column `flag` for fired category)
    runs/output_eval_explain.csv   — one row per fired flag, with the
                                     human-readable evidence

⚠️  Do **not** point `--metrics-csv` at `runs/output_security_metrics.csv`
or `--explain-csv` at `runs/output_explainability_log.csv`: those files
are appended to live by the production output_guard pipeline and use a
different column schema (`flag_name` instead of `flag`).

The analyzer is deterministic and fast (regex + entropy), so this script
runs in well under a second on the bundled 12-item set.

Aggregate summary (per category recall, FPR, average latency) is printed
to stdout — useful for quick before/after when tuning thresholds.

Usage
-----
    python evaluation/run_output_guard_batch.py \\
        --dataset datasets/output_guard_eval_set.json \\
        --metrics-csv runs/output_eval_metrics.csv \\
        --explain-csv runs/output_eval_explain.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from output_guard.output_analyzer import analyze  # noqa: E402
_DEFAULT_DATASET = _PROJECT_ROOT / "datasets" / "output_guard_eval_set.json"
# Eval-specific defaults — separate from the production paths used by
# output_guard/metrics_writer.py to prevent schema-drift overwrites.
# Column `flag` here vs `flag_name` in production; keeping them separate
# avoids corrupt mixed-schema files.
_DEFAULT_METRICS = _PROJECT_ROOT / "runs" / "output_eval_metrics.csv"
_DEFAULT_EXPLAIN = _PROJECT_ROOT / "runs" / "output_eval_explain.csv"


_METRICS_FIELDS = [
    "id", "category", "expected_decision", "decision", "score",
    "pii_triggered", "api_key_triggered", "unsafe_instruction_triggered",
    "downstream_injection_triggered", "redirect_triggered",
    "flagged_categories", "match_expected", "latency_ms", "output_chars",
]

_EXPLAIN_FIELDS = ["id", "category", "flag", "evidence"]
_FLAG_NAMES = ["pii", "api_key", "unsafe_instruction", "downstream_injection", "redirect"]


def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("items") or raw  # tolerate both {items:[...]} and bare list
    if not isinstance(items, list):
        raise ValueError(f"unexpected dataset shape in {path}")
    return items


def _flag_triggered(flags: Dict[str, Any], name: str) -> int:
    f = flags.get(name) or {}
    return 1 if bool(f.get("triggered")) else 0


def _flag_evidence(flags: Dict[str, Any], name: str) -> List[str]:
    f = flags.get(name) or {}
    evid = f.get("samples") or f.get("evidence") or []
    if isinstance(evid, list):
        return [str(x) for x in evid]
    return [str(evid)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=str(_DEFAULT_DATASET))
    ap.add_argument("--metrics-csv", default=str(_DEFAULT_METRICS))
    ap.add_argument("--explain-csv", default=str(_DEFAULT_EXPLAIN))
    args = ap.parse_args()

    items = _load_dataset(Path(args.dataset))

    metrics_path = Path(args.metrics_csv)
    explain_path = Path(args.explain_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    explain_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    total = len(items)
    by_cat: Dict[str, List[int]] = defaultdict(lambda: [0, 0])  # [hit, total]
    latencies: List[int] = []

    with metrics_path.open("w", encoding="utf-8", newline="") as fm, \
         explain_path.open("w", encoding="utf-8", newline="") as fe:
        mw = csv.DictWriter(fm, fieldnames=_METRICS_FIELDS)
        ew = csv.DictWriter(fe, fieldnames=_EXPLAIN_FIELDS)
        mw.writeheader()
        ew.writeheader()

        for it in items:
            text = it.get("text", "") or ""
            expected = (it.get("expected") or "allow").lower()
            category = it.get("category", "uncategorised")
            iid = it.get("id", "?")

            result = analyze(text)
            flags = result.flags or {}
            flagged = [k for k in _FLAG_NAMES if _flag_triggered(flags, k)]
            match = int(result.decision.lower() == expected)
            correct += match
            by_cat[category][0] += match
            by_cat[category][1] += 1
            latencies.append(int(result.latency_ms or 0))

            mw.writerow({
                "id": iid,
                "category": category,
                "expected_decision": expected,
                "decision": result.decision,
                "score": f"{result.score:.4f}",
                "pii_triggered": _flag_triggered(flags, "pii"),
                "api_key_triggered": _flag_triggered(flags, "api_key"),
                "unsafe_instruction_triggered": _flag_triggered(flags, "unsafe_instruction"),
                "downstream_injection_triggered": _flag_triggered(flags, "downstream_injection"),
                "redirect_triggered": _flag_triggered(flags, "redirect"),
                "flagged_categories": "|".join(sorted(flagged)),
                "match_expected": match,
                "latency_ms": int(result.latency_ms or 0),
                "output_chars": int(result.output_chars or len(text)),
            })

            for flag_name in _FLAG_NAMES:
                if not _flag_triggered(flags, flag_name):
                    continue
                ev = " | ".join(_flag_evidence(flags, flag_name))
                ew.writerow({
                    "id": iid,
                    "category": category,
                    "flag": flag_name,
                    "evidence": ev[:300],
                })

    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    print(f"[output_guard] wrote {metrics_path}")
    print(f"[output_guard] wrote {explain_path}")
    print(f"[output_guard] accuracy {correct}/{total} = {correct / total:.2%}  avg_latency_ms={avg_lat:.1f}")
    print("[output_guard] per-category:")
    for cat in sorted(by_cat.keys()):
        hit, tot = by_cat[cat]
        print(f"  {cat:>22s}  {hit}/{tot}  ({hit / tot:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
