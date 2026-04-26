"""evaluation/build_rag_artefacts.py
Derive three RAG eval artefacts from on-disk run data (eval-only paths —
distinct from the production live writer in `rag_guard/metrics_writer.py`
to prevent schema-drift overwrites):

    runs/rag_eval_final.csv         — canonical hybrid run, one row/doc
                                       (12-column eval schema)
    runs/rag_latency_optimized.csv  — best latency config that still
                                       meets recall ≥ 0.85
    runs/rag_eval_explain.csv       — chunk-level breakdown for every
                                       poisoned doc (debug aid)

⚠️  Do **not** repoint these to `runs/rag_final_metrics.csv` or
`runs/rag_explainability_log.csv`: those files are appended to live by
the production rag_guard pipeline and use a wider 19-column schema
(`run_id`, `target_id`, `route_*`, `judge_weight`, …) that is not a
superset of the eval schema.

These CSVs are referenced from `reports/final_evaluation.md` and the
dashboard. The script is pure CSV → CSV — no LLM, no embeddings.

Sources
-------
    runs/rag_advanced_hybrid_metrics.csv   (per-doc, includes chunk_breakdown JSON)
    runs/chunking_sweep_metrics.csv        (per-doc per-config)
    runs/chunking_sweep_summary.csv        (per-config aggregate)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_RUNS = _PROJECT_ROOT / "runs"

_HYBRID = _RUNS / "rag_advanced_hybrid_metrics.csv"
_SWEEP = _RUNS / "chunking_sweep_metrics.csv"
_SWEEP_SUMMARY = _RUNS / "chunking_sweep_summary.csv"

# Eval-specific outputs — intentionally different from the production paths
# written by rag_guard/metrics_writer.py (rag_final_metrics.csv /
# rag_explainability_log.csv) to prevent schema-drift overwrites.
_OUT_FINAL = _RUNS / "rag_eval_final.csv"
_OUT_LATENCY = _RUNS / "rag_latency_optimized.csv"
_OUT_EXPLAIN = _RUNS / "rag_eval_explain.csv"

_RECALL_FLOOR = 0.85  # latency-optimised config must clear this recall


# ---------------------------------------------------------------------------
# rag_final_metrics.csv
# ---------------------------------------------------------------------------
_FINAL_FIELDS = [
    "doc_id", "is_poisoned", "detected", "status",
    "embedding_score", "judge_score", "combined_score", "risk_score", "decision",
    "poison_type", "poison_technique", "latency_ms",
]


def build_final() -> Dict[str, Any]:
    if not _HYBRID.exists():
        raise SystemExit(f"missing source: {_HYBRID}")
    with _HYBRID.open(encoding="utf-8") as f, _OUT_FINAL.open("w", encoding="utf-8", newline="") as out:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(out, fieldnames=_FINAL_FIELDS)
        writer.writeheader()
        n = 0
        for row in reader:
            writer.writerow({k: row.get(k, "") for k in _FINAL_FIELDS})
            n += 1
    return {"path": _OUT_FINAL, "rows": n}


# ---------------------------------------------------------------------------
# rag_latency_optimized.csv
# ---------------------------------------------------------------------------
def _pick_latency_config() -> Optional[str]:
    """Return the chunking config with the lowest avg latency that meets recall floor."""
    if not _SWEEP_SUMMARY.exists():
        return None
    best = None
    with _SWEEP_SUMMARY.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                recall = float(row.get("recall", "0") or 0)
                lat = float(row.get("avg_latency_ms_per_query", "0") or 0)
            except ValueError:
                continue
            if recall < _RECALL_FLOOR:
                continue
            if best is None or lat < best[1]:
                best = (row["config"], lat, recall)
    return best[0] if best else None


def build_latency_optimised() -> Dict[str, Any]:
    if not _SWEEP.exists():
        raise SystemExit(f"missing source: {_SWEEP}")
    config = _pick_latency_config()
    if config is None:
        raise SystemExit(
            f"no chunking config met recall ≥ {_RECALL_FLOOR}; "
            f"check {_SWEEP_SUMMARY}"
        )
    with _SWEEP.open(encoding="utf-8") as f, _OUT_LATENCY.open("w", encoding="utf-8", newline="") as out:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(out, fieldnames=reader.fieldnames or [])
        writer.writeheader()
        n = 0
        for row in reader:
            if row.get("config") == config:
                writer.writerow(row)
                n += 1
    return {"path": _OUT_LATENCY, "config": config, "rows": n}


# ---------------------------------------------------------------------------
# rag_explainability_log.csv  — flatten chunk_breakdown JSON
# ---------------------------------------------------------------------------
_EXPLAIN_FIELDS = [
    "doc_id", "is_poisoned", "status", "decision",
    "chunk_idx", "chunk_text_preview", "chunk_judge_score",
    "chunk_latency_ms", "chunk_error",
]


def build_explainability() -> Dict[str, Any]:
    if not _HYBRID.exists():
        raise SystemExit(f"missing source: {_HYBRID}")
    rows_written = 0
    docs_with_breakdown = 0
    with _HYBRID.open(encoding="utf-8") as f, _OUT_EXPLAIN.open("w", encoding="utf-8", newline="") as out:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(out, fieldnames=_EXPLAIN_FIELDS)
        writer.writeheader()
        for row in reader:
            cb_raw = (row.get("chunk_breakdown") or "").strip()
            if not cb_raw or cb_raw in ("null", "[]", "None"):
                continue
            try:
                breakdown = json.loads(cb_raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(breakdown, list):
                continue
            docs_with_breakdown += 1
            for chunk in breakdown:
                if not isinstance(chunk, dict):
                    continue
                writer.writerow({
                    "doc_id": row.get("doc_id", ""),
                    "is_poisoned": row.get("is_poisoned", ""),
                    "status": row.get("status", ""),
                    "decision": row.get("decision", ""),
                    "chunk_idx": chunk.get("idx", ""),
                    "chunk_text_preview": (chunk.get("text_preview") or "")[:200],
                    "chunk_judge_score": chunk.get("judge_score", ""),
                    "chunk_latency_ms": chunk.get("latency_ms", ""),
                    "chunk_error": chunk.get("error", "") or "",
                })
                rows_written += 1
    return {
        "path": _OUT_EXPLAIN,
        "docs_with_breakdown": docs_with_breakdown,
        "chunk_rows": rows_written,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    final = build_final()
    print(f"[rag_final] wrote {final['path']} ({final['rows']} rows)")
    lat = build_latency_optimised()
    print(f"[rag_latency_opt] wrote {lat['path']} (config={lat['config']}, {lat['rows']} rows)")
    expl = build_explainability()
    print(f"[rag_explain] wrote {expl['path']} "
          f"({expl['docs_with_breakdown']} docs × {expl['chunk_rows']} chunk rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
