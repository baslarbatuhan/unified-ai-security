"""
evaluation/prompt_guard_stability.py
====================================
Task 6: measure PromptGuardPipeline stability on a benign set.

Metrics:
  - FP rate under the live (adaptive) threshold config from secure_balanced.yaml
  - FP rate with the deobfuscator disabled (side-effect check)
  - Up to 10 FP example prompts with their risk scores

This is a measurement-only script. It does NOT change thresholds or detectors.

Input:
    datasets/injection_prompts/injection_dataset_v1.csv  (label==0 rows)

Output:
    runs/prompt_guard_stability.json
    reports/prompt_guard_stability.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except Exception:
    pass

from prompt_guard import pipeline as pg_pipeline_mod  # noqa: E402
from prompt_guard.pipeline import PromptGuardPipeline  # noqa: E402


DATASET_CSV = _PROJECT_ROOT / "datasets" / "injection_prompts" / "injection_dataset_v1.csv"
RUNS_DIR = _PROJECT_ROOT / "runs"
REPORTS_DIR = _PROJECT_ROOT / "reports"


def _load_benign() -> List[str]:
    rows = list(csv.reader(DATASET_CSV.open(encoding="utf-8")))
    return [r[0] for r in rows[1:] if r[-1] == "0"]


def _thresholds_from_yaml() -> Dict[str, float]:
    """Read secure_balanced.yaml prompt_guard.adaptive_thresholds if present."""
    try:
        import yaml  # type: ignore
        y = yaml.safe_load((_PROJECT_ROOT / "configs" / "secure_balanced.yaml").read_text())
        pg = ((y.get("modules") or {}).get("prompt_guard") or {})
        return {
            "semantic_threshold": float(pg.get("semantic_threshold", 0.65)),
            "short": float((pg.get("adaptive_thresholds") or {}).get("short", 0.55)),
            "medium": float((pg.get("adaptive_thresholds") or {}).get("medium", 0.60)),
            "long": float((pg.get("adaptive_thresholds") or {}).get("long", 0.65)),
        }
    except Exception:
        return {"semantic_threshold": 0.65, "short": 0.55, "medium": 0.60, "long": 0.65}


def _run_pipeline(pipe: PromptGuardPipeline, benign: List[str]) -> Dict[str, Any]:
    """Run benign prompts and score under 3 FP criteria:
      - block: decision == 'block'           (hard FP — user request denied)
      - non_allow: decision != 'allow'       (soft FP — includes sanitize/flag)
      - injection: is_injection == True      (native semantic-threshold crossing)
    The prompt_guard module contributes only ~0.30 of the fusion score, so a
    "sanitize" decision does NOT necessarily translate to a user-visible block
    at the gateway. Report all three to make the trade-off explicit.
    """
    counts = {"block": 0, "non_allow": 0, "injection": 0}
    examples: Dict[str, List[Dict[str, Any]]] = {"block": [], "non_allow": []}
    total_latency = 0
    risks: List[float] = []
    for p in benign:
        t0 = time.time()
        r = pipe.run(p)
        total_latency += int((time.time() - t0) * 1000)
        decision = r.risk.decision if r.risk else "allow"
        risk = r.risk.risk_score if r.risk else 0.0
        is_injection = bool(r.is_injection)
        risks.append(risk)

        if decision == "block":
            counts["block"] += 1
            if len(examples["block"]) < 10:
                examples["block"].append(
                    {"prompt": p[:160], "risk": round(risk, 4), "decision": decision}
                )
        if decision != "allow":
            counts["non_allow"] += 1
            if len(examples["non_allow"]) < 10:
                examples["non_allow"].append(
                    {"prompt": p[:160], "risk": round(risk, 4), "decision": decision}
                )
        if is_injection:
            counts["injection"] += 1

    n = max(1, len(benign))
    return {
        "total_benign": len(benign),
        "fp_block": counts["block"],
        "fp_non_allow": counts["non_allow"],
        "fp_injection_flag": counts["injection"],
        "fp_rate_block": round(counts["block"] / n, 4),
        "fp_rate_non_allow": round(counts["non_allow"] / n, 4),
        "fp_rate_injection": round(counts["injection"] / n, 4),
        "avg_risk": round(sum(risks) / n, 4),
        "max_risk": round(max(risks) if risks else 0.0, 4),
        "fp_examples_block": examples["block"],
        "fp_examples_non_allow": examples["non_allow"],
        "avg_latency_ms": int(total_latency / n),
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Prompt Guard stability — FP check and (optional) regression metrics."
    )
    ap.add_argument(
        "--dataset",
        choices=["benign", "regression"],
        default="benign",
        help=(
            "'benign' (default): legacy FP-only mode using injection_dataset_v1.csv "
            "label==0 rows. 'regression': use datasets/prompt_regression_set.json "
            "(curated benign + attack cases) to compute precision/recall/F1/FPR."
        ),
    )
    ap.add_argument(
        "--regression-path",
        default=None,
        help="Override path for the regression JSON (defaults to datasets/prompt_regression_set.json).",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Regression mode only: treat only decision=='block' as positive. "
             "Default counts any non-allow decision as positive (module-level).",
    )
    return ap.parse_args(argv)


def _run_regression_mode(args: argparse.Namespace) -> int:
    """Regression mode: curated set → precision/recall/F1/FPR + per-category."""
    sys.path.insert(0, str(_PROJECT_ROOT))  # ensure `datasets.*` importable
    from datasets.dataset_loaders import (  # noqa: E402
        DEFAULT_PROMPT_REGRESSION_SET, load_prompt_regression_set,
        compute_confusion, compute_metrics, per_category_breakdown,
    )

    ds_path = Path(args.regression_path) if args.regression_path else DEFAULT_PROMPT_REGRESSION_SET
    cases = load_prompt_regression_set(ds_path)
    print(f"[prompt_guard_stability] dataset=regression path={ds_path} "
          f"cases={len(cases)} (benign={sum(c.is_benign for c in cases)} "
          f"attack={sum(c.is_attack for c in cases)})")

    yaml_thr = _thresholds_from_yaml()
    pipe = PromptGuardPipeline(
        semantic_threshold=yaml_thr["semantic_threshold"],
        adaptive_tier_thresholds=(yaml_thr["short"], yaml_thr["medium"], yaml_thr["long"]),
    )

    observed: Dict[str, str] = {}
    per_case_rows: List[Dict[str, Any]] = []
    total_latency = 0
    for c in cases:
        t0 = time.time()
        r = pipe.run(c.prompt)
        lat = int((time.time() - t0) * 1000)
        total_latency += lat
        decision = r.risk.decision if r.risk else "allow"
        risk = r.risk.risk_score if r.risk else 0.0
        observed[c.id] = decision
        per_case_rows.append({
            "id": c.id,
            "category": c.category,
            "label": c.label,
            "expected": c.expected_decision,
            "observed": decision,
            "risk": round(risk, 4),
            "match": int(decision == c.expected_decision),
            "latency_ms": lat,
        })

    # Metrics under two positivity modes: strict (block only) and soft (any non-allow).
    cm_strict = compute_confusion(cases, observed, strict=True)
    m_strict = compute_metrics(cm_strict)
    cm_soft = compute_confusion(cases, observed, strict=False)
    m_soft = compute_metrics(cm_soft)
    # Use the CLI-selected mode for the category breakdown.
    per_cat = per_category_breakdown(cases, observed, strict=args.strict)

    summary = {
        "mode": "regression",
        "dataset": str(ds_path.relative_to(_PROJECT_ROOT))
                   if ds_path.is_absolute() and ds_path.is_relative_to(_PROJECT_ROOT)
                   else str(ds_path),
        "thresholds_used": yaml_thr,
        "total_cases": len(cases),
        "strict": {"confusion": cm_strict, **m_strict},
        "soft":   {"confusion": cm_soft,   **m_soft},
        "per_category": per_cat,
        "per_category_strict": args.strict,
        "avg_latency_ms": int(total_latency / max(1, len(cases))),
        "per_case": per_case_rows,
    }

    RUNS_DIR.mkdir(exist_ok=True)
    out_json = RUNS_DIR / "prompt_guard_regression.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[saved] {out_json}")

    REPORTS_DIR.mkdir(exist_ok=True)
    md = [
        "# Prompt Guard Regression — Curated Set",
        "",
        f"**Dataset:** `{summary['dataset']}` ({len(cases)} cases)",
        f"**Thresholds (YAML):** short={yaml_thr['short']}, medium={yaml_thr['medium']}, "
        f"long={yaml_thr['long']}, base={yaml_thr['semantic_threshold']}",
        "",
        "## Binary metrics (attack = positive)",
        "",
        "| positivity | precision | recall | F1 | accuracy | FPR | TP | FP | TN | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (f"| strict (block only) | {m_strict['precision']:.3f} | {m_strict['recall']:.3f} | "
         f"{m_strict['f1']:.3f} | {m_strict['accuracy']:.3f} | {m_strict['fpr']:.3f} | "
         f"{cm_strict['tp']} | {cm_strict['fp']} | {cm_strict['tn']} | {cm_strict['fn']} |"),
        (f"| soft (non-allow)    | {m_soft['precision']:.3f} | {m_soft['recall']:.3f} | "
         f"{m_soft['f1']:.3f} | {m_soft['accuracy']:.3f} | {m_soft['fpr']:.3f} | "
         f"{cm_soft['tp']} | {cm_soft['fp']} | {cm_soft['tn']} | {cm_soft['fn']} |"),
        "",
        f"**Average latency:** {summary['avg_latency_ms']} ms/case",
        "",
        "## Per-category (positivity: "
        + ("strict" if args.strict else "soft") + ")",
        "",
        "| category | n | F1 | precision | recall | FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cat, metrics in sorted(per_cat.items()):
        md.append(
            f"| {cat} | {int(metrics['total'])} | {metrics['f1']:.3f} | "
            f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['fpr']:.3f} |"
        )

    out_md = REPORTS_DIR / "prompt_guard_regression.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[saved] {out_md}")
    print(f"  strict F1={m_strict['f1']:.3f}  soft F1={m_soft['f1']:.3f}  "
          f"FPR(strict)={m_strict['fpr']:.3f}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.dataset == "regression":
        return _run_regression_mode(args)
    benign = _load_benign()
    print(f"[prompt_guard_stability] benign={len(benign)} prompts")
    yaml_thr = _thresholds_from_yaml()
    print(f"[prompt_guard_stability] YAML thresholds: {yaml_thr}")

    # --- Run A: live config (adaptive thresholds on, deobfuscator on) ---
    pipe = PromptGuardPipeline(
        semantic_threshold=yaml_thr["semantic_threshold"],
        adaptive_tier_thresholds=(yaml_thr["short"], yaml_thr["medium"], yaml_thr["long"]),
    )
    print("[run A] deobfuscator=ON, adaptive=ON …")
    run_a = _run_pipeline(pipe, benign)
    print(f"  block={run_a['fp_block']}/{run_a['total_benign']} ({run_a['fp_rate_block']:.3f}) "
          f"non_allow={run_a['fp_non_allow']} inj_flag={run_a['fp_injection_flag']} "
          f"avg_risk={run_a['avg_risk']:.3f} latency={run_a['avg_latency_ms']}ms/prompt")

    # --- Run B: deobfuscator neutralised (passthrough) ---
    orig_report = pg_pipeline_mod.get_deobfuscation_report

    def _passthrough(text: str) -> Dict:
        return {"deobfuscated": text, "changed": False, "changes": []}

    pg_pipeline_mod.get_deobfuscation_report = _passthrough  # type: ignore[assignment]
    try:
        print("[run B] deobfuscator=OFF (passthrough) …")
        run_b = _run_pipeline(pipe, benign)
        print(f"  block={run_b['fp_block']}/{run_b['total_benign']} ({run_b['fp_rate_block']:.3f}) "
              f"non_allow={run_b['fp_non_allow']} inj_flag={run_b['fp_injection_flag']} "
              f"avg_risk={run_b['avg_risk']:.3f} latency={run_b['avg_latency_ms']}ms/prompt")
    finally:
        pg_pipeline_mod.get_deobfuscation_report = orig_report  # type: ignore[assignment]

    delta_block = round(run_a["fp_rate_block"] - run_b["fp_rate_block"], 4)
    delta_non_allow = round(run_a["fp_rate_non_allow"] - run_b["fp_rate_non_allow"], 4)
    summary = {
        "dataset": str(DATASET_CSV.relative_to(_PROJECT_ROOT)),
        "thresholds_used": yaml_thr,
        "deobfuscator_on": run_a,
        "deobfuscator_off": run_b,
        "deobfuscator_fp_delta_block": delta_block,
        "deobfuscator_fp_delta_non_allow": delta_non_allow,
        "notes": (
            "FP is reported under 3 criteria (see keys fp_rate_*):\n"
            "  - block: decision=='block' (hard user-visible block) — the strict thesis metric.\n"
            "  - non_allow: decision in {sanitize, flag, block} (soft — includes downgrade actions).\n"
            "  - injection: is_injection==True (native semantic-threshold crossing).\n"
            "Fusion context: prompt_guard contributes ~0.30 to the gateway score, "
            "so a 'sanitize' decision usually does NOT translate to a user-visible "
            "block end-to-end. The block metric is the primary stability signal."
        ),
    }

    RUNS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    out_json = RUNS_DIR / "prompt_guard_stability.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[saved] {out_json}")

    # Markdown report
    md = [
        "# Prompt Guard Stability — Benign FP Check",
        "",
        f"**Dataset:** `{summary['dataset']}` ({run_a['total_benign']} benign prompts, label==0)",
        f"**Thresholds (YAML):** short={yaml_thr['short']}, medium={yaml_thr['medium']}, long={yaml_thr['long']}, base={yaml_thr['semantic_threshold']}",
        "",
        "## Results",
        "",
        "FP rate under three criteria (see Methodology below):",
        "",
        "| Configuration | block | non_allow | injection_flag | avg_risk | latency |",
        "|---|---:|---:|---:|---:|---:|",
        (f"| Deobfuscator ON (live) | {run_a['fp_rate_block']:.3f} "
         f"({run_a['fp_block']}/{run_a['total_benign']}) | {run_a['fp_rate_non_allow']:.3f} "
         f"| {run_a['fp_rate_injection']:.3f} | {run_a['avg_risk']:.3f} | {run_a['avg_latency_ms']}ms |"),
        (f"| Deobfuscator OFF       | {run_b['fp_rate_block']:.3f} "
         f"({run_b['fp_block']}/{run_b['total_benign']}) | {run_b['fp_rate_non_allow']:.3f} "
         f"| {run_b['fp_rate_injection']:.3f} | {run_b['avg_risk']:.3f} | {run_b['avg_latency_ms']}ms |"),
        (f"| **Δ (ON − OFF)**       | {delta_block:+.3f} | {delta_non_allow:+.3f} | — | — | — |"),
        "",
        "## FP examples — decision==block (user-visible FP, deobfuscator ON)",
        "",
    ]
    if run_a["fp_examples_block"]:
        md.append("| risk | decision | prompt |")
        md.append("|---:|---|---|")
        for ex in run_a["fp_examples_block"]:
            safe = ex["prompt"].replace("|", "\\|")
            md.append(f"| {ex['risk']:.3f} | {ex['decision']} | {safe} |")
    else:
        md.append("_None — no benign prompt was hard-blocked._")
    md += [
        "",
        "## FP examples — decision != allow (soft FP, deobfuscator ON, first 10)",
        "",
        "| risk | decision | prompt |",
        "|---:|---|---|",
    ]
    for ex in run_a["fp_examples_non_allow"]:
        safe = ex["prompt"].replace("|", "\\|")
        md.append(f"| {ex['risk']:.3f} | {ex['decision']} | {safe} |")

    md += [
        "",
        "## Methodology",
        "",
        "- **block** FP: standalone prompt_guard decides `block`. This is the user-visible stability metric.",
        "- **non_allow** FP: decision is `sanitize`, `flag`, or `block`. The module downgrades but does not necessarily deny; under fusion weighting (prompt_guard ≈ 0.30), most of these still `allow` end-to-end.",
        "- **injection_flag** FP: `is_injection=True` — the semantic similarity exceeded the adaptive threshold. The rawest possible signal.",
        "- No thresholds or detectors were changed; this is a baseline snapshot.",
        "",
        "## Interpretation",
        "",
        f"- Hard block FP rate: **{run_a['fp_rate_block']*100:.1f}%** ({run_a['fp_block']}/{run_a['total_benign']}) under the live config.",
        f"- The deobfuscator moves the block FP count by {run_a['fp_block'] - run_b['fp_block']:+d} (Δ rate = {delta_block:+.3f}).",
        f"- Average standalone prompt_guard risk on benign prompts is {run_a['avg_risk']:.3f} (max {run_a['max_risk']:.3f}), so most benign requests fall in the 'sanitize' band (0.30–0.60) rather than block.",
        "- The high `non_allow` rate reflects an aggressive adaptive threshold (0.55 for short prompts). Because fusion weights prompt_guard at 0.30, this rarely produces a user-visible denial — but it does suggest the thesis should distinguish standalone-module FP from gateway FP when reporting usability.",
    ]
    out_md = REPORTS_DIR / "prompt_guard_stability.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[saved] {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
