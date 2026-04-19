"""
evaluation/judge_determinism.py
===============================
Task 3: measure LLM judge determinism under fixed seed + temperature=0.

Runs each poisoned document from the advanced dataset N times through the
full document (non-chunked) analyze path, records per-call judge_score,
and emits variance statistics + an optional median-of-3 comparison.

Outputs:
  runs/judge_determinism_metrics.csv   — per (doc, trial) rows
  runs/judge_determinism_summary.csv   — per-doc variance summary
  reports/judge_determinism.md         — human-readable digest
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUNS_DIR = _PROJECT_ROOT / "runs"
_REPORTS_DIR = _PROJECT_ROOT / "reports"
_DATASET = _PROJECT_ROOT / "datasets" / "poisoned_corpus" / "advanced_poison_samples.json"

sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except Exception:
    pass

from rag_guard.llm_judge import LLMJudge  # noqa: E402


def _verify_temperature_zero(judge: LLMJudge) -> Dict[str, Any]:
    """Sanity-check: confirm judge's request payload uses temperature=0 and seed=42."""
    # Inspect the outbound payload by monkey-patching requests.post for a no-op.
    import requests  # type: ignore

    captured: Dict[str, Any] = {}
    orig_post = requests.post

    class _FakeResp:
        status_code = 200
        text = ""

        def json(self):
            return {"message": {"content": '{"risk_score": 0.0, "reason": "probe"}'}}

    def _patched_post(url, *a, **kw):  # noqa: ARG001
        captured["url"] = url
        captured["json"] = kw.get("json") or (a[0] if a else None)
        return _FakeResp()

    requests.post = _patched_post  # type: ignore[assignment]
    try:
        try:
            judge.analyze("probe", doc_id="probe", user_query="probe")
        except Exception:
            pass  # even if parsing fails, we only care about the captured payload
    finally:
        requests.post = orig_post  # type: ignore[assignment]

    payload = captured.get("json") or {}
    opts = payload.get("options") or {}
    return {
        "url": captured.get("url"),
        "temperature": opts.get("temperature"),
        "seed": opts.get("seed"),
        "payload_options": opts,
    }


def _run_trials(judge: LLMJudge, docs: List[Dict], trials: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for doc in docs:
        doc_id = doc.get("doc_id", "unknown")
        content = doc.get("content", "")
        tq = doc.get("target_query") or "general query"
        for t in range(trials):
            t0 = time.time()
            r = judge.analyze(content=content, doc_id=doc_id, user_query=tq)
            latency = int((time.time() - t0) * 1000)
            rows.append({
                "doc_id": doc_id,
                "trial": t,
                "judge_score": round(r.judge_score, 4),
                "latency_ms": latency,
                "error": r.error or "",
                "reason": (r.explanation or "")[:160],
            })
            print(f"  {doc_id:<18s} trial={t} score={r.judge_score:.3f} ({latency}ms)")
    return rows


def _summarise(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_doc: Dict[str, List[float]] = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], []).append(r["judge_score"])
    out: List[Dict[str, Any]] = []
    for doc_id, scores in by_doc.items():
        mean = statistics.fmean(scores)
        std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        out.append({
            "doc_id": doc_id,
            "n": len(scores),
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "cv": round(std / mean, 4) if mean > 0 else 0.0,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5,
                    help="Trials per doc (default 5).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max poisoned docs to test (0 = all).")
    args = ap.parse_args()

    data = json.loads(_DATASET.read_text(encoding="utf-8"))
    poisoned = [d for d in data["documents"] if d["is_poisoned"]]
    if args.limit > 0:
        poisoned = poisoned[: args.limit]

    judge = LLMJudge()

    # ---- Step 1: verify temperature=0, seed=42 in outbound payload ----
    print("[determinism] verifying request payload uses temperature=0, seed=42 …")
    probe = _verify_temperature_zero(judge)
    temp_ok = probe.get("temperature") == 0 or probe.get("temperature") == 0.0
    seed_ok = probe.get("seed") == 42
    print(f"  captured options: {probe.get('payload_options')}")
    print(f"  temperature==0? {temp_ok}  seed==42? {seed_ok}")
    assert temp_ok, f"temperature is not 0: {probe.get('temperature')}"

    # ---- Step 2: run trials ----
    print(f"[determinism] running {len(poisoned)} poisoned docs × {args.trials} trials …")
    t0 = time.time()
    rows = _run_trials(judge, poisoned, args.trials)
    elapsed = time.time() - t0
    print(f"[determinism] {len(rows)} calls in {elapsed/60:.1f} min")

    # ---- Step 3: summarise ----
    summary = _summarise(rows)
    dataset_mean_std = round(
        statistics.fmean(s["std"] for s in summary), 4
    ) if summary else 0.0
    docs_above_005 = [s for s in summary if s["std"] > 0.05]
    top3_var = sorted(summary, key=lambda s: -s["std"])[:3]

    _RUNS_DIR.mkdir(exist_ok=True)
    metrics_path = _RUNS_DIR / "judge_determinism_metrics.csv"
    summary_path = _RUNS_DIR / "judge_determinism_summary.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # ---- Step 4: report ----
    _REPORTS_DIR.mkdir(exist_ok=True)
    md = [
        "# LLM Judge Determinism",
        "",
        f"**Model:** `{judge.model}` (fallback `{judge.fallback_model}`)  ",
        f"**Dataset:** 15 advanced poisoned docs × {args.trials} trials each  ",
        f"**Payload options:** `{probe.get('payload_options')}`  ",
        f"**temperature==0:** {temp_ok} · **seed==42:** {seed_ok}",
        "",
        "## Dataset-level variance",
        "",
        f"- Mean within-doc std: **{dataset_mean_std:.4f}**",
        f"- Docs with std > 0.05: **{len(docs_above_005)} / {len(summary)}**",
        "- Interpretation: a mean std near 0 means the judge is near-deterministic "
        "at the claimed seed/temperature; >0.05 indicates sampling noise is meaningful.",
        "",
        "## Top-3 highest-variance docs",
        "",
        "| doc_id | n | mean | std | min | max | cv |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in top3_var:
        md.append(
            f"| {s['doc_id']} | {s['n']} | {s['mean']:.3f} | {s['std']:.4f} | "
            f"{s['min']:.3f} | {s['max']:.3f} | {s['cv']:.3f} |"
        )

    md += [
        "",
        "## Per-doc summary",
        "",
        "| doc_id | n | mean | std | min | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in sorted(summary, key=lambda s: s["doc_id"]):
        md.append(
            f"| {s['doc_id']} | {s['n']} | {s['mean']:.3f} | {s['std']:.4f} | "
            f"{s['min']:.3f} | {s['max']:.3f} |"
        )

    md += [
        "",
        "## Reading the result",
        "",
        f"- If `std > 0.05` for a doc whose mean sits near the 0.45 poison_threshold, "
        "it can flip TP↔FN across runs. Ablation and Task-2 sweep metrics inherit "
        "this noise — report confidence intervals, not raw single-run deltas.",
        f"- Dataset-level mean std of {dataset_mean_std:.4f} "
        f"{'exceeds 0.05 — consider median-of-3 voting' if dataset_mean_std > 0.05 else 'is below 0.05 — single-call mode is stable enough'}.",
    ]
    (_REPORTS_DIR / "judge_determinism.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[saved] {metrics_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {_REPORTS_DIR / 'judge_determinism.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
