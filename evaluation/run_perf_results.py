"""evaluation/run_perf_results.py
Per-module performance summary derived from
`runs/gateway_attack_results.csv`.

Each row contains prompt_latency_ms, rag_latency_ms, agency_latency_ms,
fusion_latency_ms and the total latency_ms. We aggregate to mean / median /
p95 / max per module across all attacks plus per-attack-type slices.

Output:
    runs/perf_results.csv   (long format: module, attack_type, n, mean, p50, p95, max)
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parent.parent
_SOURCE = _ROOT / "runs" / "gateway_attack_results.csv"
_OUT = _ROOT / "runs" / "perf_results.csv"

MODULES = [
    ("prompt_guard", "prompt_latency_ms"),
    ("rag_guard", "rag_latency_ms"),
    ("output_agency", "agency_latency_ms"),
    ("fusion", "fusion_latency_ms"),
    ("end_to_end", "latency_ms"),
]


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main() -> Path:
    if not _SOURCE.exists():
        raise SystemExit(f"missing source: {_SOURCE}")

    with _SOURCE.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("attack_type") and r["attack_type"] != "attack_type"]

    # buckets[(module, attack_type)] = [latency, ...]; ALL bucket aggregates
    buckets: Dict[tuple, List[float]] = defaultdict(list)
    for row in rows:
        atype = row["attack_type"]
        for mod, col in MODULES:
            v = _safe_float(row.get(col, "0"))
            # 0-ms entries usually mean "module did not run" for non-prompt modules.
            # Keep them in end_to_end and fusion (always run); skip for per-module stats.
            if mod in ("rag_guard", "output_agency") and v == 0:
                continue
            buckets[(mod, atype)].append(v)
            buckets[(mod, "ALL")].append(v)

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["module", "attack_type", "n", "mean_ms", "p50_ms", "p95_ms", "max_ms"])
        for mod, _col in MODULES:
            keys = sorted([k for k in buckets if k[0] == mod and k[1] != "ALL"], key=lambda k: k[1])
            for k in keys:
                xs = buckets[k]
                if not xs:
                    continue
                w.writerow([
                    mod, k[1], len(xs),
                    f"{sum(xs) / len(xs):.1f}",
                    f"{_percentile(xs, 0.50):.1f}",
                    f"{_percentile(xs, 0.95):.1f}",
                    f"{max(xs):.1f}",
                ])
            xs = buckets.get((mod, "ALL"), [])
            if xs:
                w.writerow([
                    mod, "ALL", len(xs),
                    f"{sum(xs) / len(xs):.1f}",
                    f"{_percentile(xs, 0.50):.1f}",
                    f"{_percentile(xs, 0.95):.1f}",
                    f"{max(xs):.1f}",
                ])

    print(f"[perf] wrote {_OUT}  ({len(rows)} rows × {len(MODULES)} modules)")
    return _OUT


if __name__ == "__main__":
    main()
