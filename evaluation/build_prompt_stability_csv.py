"""evaluation/build_prompt_stability_csv.py
Flatten ``runs/prompt_guard_stability.json`` into a spec-named CSV view.

The JSON has nested per-mode (deobfuscator_on / deobfuscator_off) blocks
with three FP definitions (block / non_allow / injection). The CSV emits
one row per (mode, fp_definition) pair so downstream analysis tools and
the dashboard can read the key numbers without parsing JSON.

    runs/prompt_stability_check.csv
        mode, fp_criterion, total_benign, fp_count, fp_rate,
        avg_risk, max_risk, avg_latency_ms

This is pure JSON → CSV; no model calls.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "runs" / "prompt_guard_stability.json"
_OUT = _ROOT / "runs" / "prompt_stability_check.csv"

_FIELDS = [
    "mode", "fp_criterion", "total_benign", "fp_count",
    "fp_rate", "avg_risk", "max_risk", "avg_latency_ms",
]

_CRITERIA = [
    ("block", "fp_block", "fp_rate_block"),
    ("non_allow", "fp_non_allow", "fp_rate_non_allow"),
    ("injection", "fp_injection_flag", "fp_rate_injection"),
]


def main() -> int:
    if not _SRC.exists():
        raise SystemExit(f"missing source: {_SRC}")
    data = json.loads(_SRC.read_text(encoding="utf-8"))
    rows = []
    for mode in ("deobfuscator_on", "deobfuscator_off"):
        block = data.get(mode) or {}
        if not block:
            continue
        for label, count_key, rate_key in _CRITERIA:
            rows.append({
                "mode": mode,
                "fp_criterion": label,
                "total_benign": block.get("total_benign", ""),
                "fp_count": block.get(count_key, ""),
                "fp_rate": block.get(rate_key, ""),
                "avg_risk": block.get("avg_risk", ""),
                "max_risk": block.get("max_risk", ""),
                "avg_latency_ms": block.get("avg_latency_ms", ""),
            })
    with _OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[prompt_stability] wrote {_OUT} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
