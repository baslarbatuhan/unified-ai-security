"""scripts/generate_thesis_report.py
=========================================
Automated evaluation report — Hafta 15.

Single command:
    .venv/bin/python scripts/generate_thesis_report.py [--n 20] [--out reports/thesis_evaluation.md]

Walks `runs/_registry.jsonl` (newest first, capped at --n), aggregates
the per-run results.csv files, and emits:

    reports/<out>.md
    reports/<out>.charts/
        attack_distribution.png
        per_module_risk.png
        latency_histogram.png
        decision_breakdown.png
        confusion_block.png      (only when GT is available)

The markdown report is structured to drop directly into a thesis
appendix: title page → executive summary → per-suite breakdown →
latency analysis → recommendations → raw appendix.

Design notes
------------
* Pure-fn data layer reuses `dashboard.lib.recommendations` so the
  Streamlit dashboard and the thesis report share the same security
  score formula (single source of truth).
* matplotlib backend pinned to Agg in `_set_headless_backend()` so the
  script runs inside Docker without an X server.
* PDF generation deliberately omitted — markdown + PNG charts converts
  trivially via pandoc / vscode / typora. Adding pdfkit/weasyprint
  here would pull in heavyweight system deps; users that need PDF can
  pipe `pandoc out.md -o out.pdf`.
* `--baseline` flag enables before/after diff against a pinned run id
  (useful for regression reports). When absent, the report stands alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.run_manifest import RESULTS_FILENAME, read_registry  # noqa: E402

# These two helpers stay shared with the dashboard — moving them here
# would duplicate the security-score formula and let the two surfaces
# drift apart. Use the same code path on purpose.
from dashboard.lib.recommendations import (  # noqa: E402
    composite_security_score,
    generate_recommendations,
)


_RUNS_DIR = _PROJECT_ROOT / "runs"
_LEGACY_AGG_CSV = _RUNS_DIR / "external_eval_results.csv"


# ---------------------------------------------------------------------------
# Headless matplotlib bootstrap
# ---------------------------------------------------------------------------
def _set_headless_backend() -> None:
    """Pin Agg before importing pyplot so Docker / CI without an X
    server still renders to PNG."""
    import matplotlib
    matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_per_run_rows(entry: Dict[str, Any]):
    """Prefer the per-run results.csv; fall back to filtering the
    legacy aggregate. Returns a pandas DataFrame (may be empty)."""
    import pandas as pd
    rid = entry.get("run_id") or ""
    if not rid:
        return pd.DataFrame()
    per = _RUNS_DIR / rid / RESULTS_FILENAME
    if per.exists():
        try:
            return pd.read_csv(per)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
    if _LEGACY_AGG_CSV.exists():
        try:
            full = pd.read_csv(_LEGACY_AGG_CSV)
            return full[full["run_id"].astype(str) == rid].reset_index(drop=True)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
    return pd.DataFrame()


def _load_runs(limit: int) -> List[Dict[str, Any]]:
    return read_registry(_RUNS_DIR, limit=limit)


# ---------------------------------------------------------------------------
# Aggregation — same shape `1_home.py` builds, deduplicated here so the
# report can be generated standalone (no Streamlit imports).
# ---------------------------------------------------------------------------
def _aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    import pandas as pd
    if not runs:
        return {"n_recent_runs": 0, "n_rows": 0}
    frames = []
    for r in runs:
        df = _load_per_run_rows(r)
        if not df.empty:
            frames.append(df)
    if not frames:
        return {"n_recent_runs": len(runs), "n_rows": 0}
    df = pd.concat(frames, ignore_index=True)
    total = len(df)

    dec = df.get("gateway_decision", pd.Series(dtype=str)).astype(str).str.lower()
    band = df.get("gateway_decision_band", pd.Series(dtype=str)).astype(str).str.lower()
    miss = pd.to_numeric(df.get("gateway_miss", pd.Series(dtype=int)),
                         errors="coerce").fillna(0).astype(int)

    block_count = int((dec == "block").sum())
    sanitize_count = int((dec == "sanitize").sum())
    allow_count = int((dec == "allow").sum())
    miss_count = int(miss.sum())

    exp = df.get("expected_decision", pd.Series(dtype=str)).astype(str).str.lower()
    tp = int(((exp == "block") & (dec == "block")).sum())
    fn = int(((exp == "block") & (dec != "block")).sum())
    fp = int(((exp != "block") & (dec == "block")).sum())
    tn = int(((exp != "block") & (dec != "block")).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0

    lat = pd.to_numeric(df.get("gateway_latency_ms", pd.Series(dtype=float)),
                        errors="coerce").dropna()
    avg_latency = float(lat.mean()) if not lat.empty else 0.0
    p50 = float(lat.quantile(0.5)) if not lat.empty else 0.0
    p95 = float(lat.quantile(0.95)) if not lat.empty else 0.0
    latency_breach_rate = 0.0
    if len(lat) >= 5:
        sigma_threshold = lat.mean() + 2 * (lat.std(ddof=0) or 0.0)
        latency_breach_rate = float((lat > sigma_threshold).mean())

    suite_dist = (
        df.get("suite", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
    )
    mod_means = {}
    for col, label in (
        ("gateway_prompt_score", "prompt_guard"),
        ("gateway_rag_score", "rag_guard"),
        ("gateway_agency_score", "output_agency"),
    ):
        if col in df.columns:
            mod_means[label] = float(pd.to_numeric(df[col], errors="coerce").mean() or 0.0)

    return {
        "n_recent_runs": len(runs),
        "n_rows": total,
        "total": total,
        "block_count": block_count,
        "sanitize_count": sanitize_count,
        "allow_count": allow_count,
        "miss_count": miss_count,
        "miss_rate": (miss_count / total) if total else 0.0,
        "block_rate": (block_count / total) if total else 0.0,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "fp_rate": fp_rate,
        "avg_latency_ms": avg_latency, "p50_latency_ms": p50, "p95_latency_ms": p95,
        "latency_breach_rate": latency_breach_rate,
        "suite_distribution": suite_dist,
        "module_mean_scores": mod_means,
        "raw_df": df,
    }


# ---------------------------------------------------------------------------
# Chart rendering
# ---------------------------------------------------------------------------
def _render_attack_distribution(agg, charts_dir: Path) -> Optional[Path]:
    import matplotlib.pyplot as plt
    dist = agg.get("suite_distribution") or {}
    if not dist:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(dist.keys())
    sizes = list(dist.values())
    bars = ax.bar(labels, sizes, color=["#3b82f6", "#a855f7", "#ec4899", "#f97316"][:len(labels)])
    ax.set_title("Attack suite distribution (recent runs)")
    ax.set_ylabel("Cases")
    for b, n in zip(bars, sizes):
        ax.annotate(str(n), xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p = charts_dir / "attack_distribution.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _render_per_module_risk(agg, charts_dir: Path) -> Optional[Path]:
    import matplotlib.pyplot as plt
    means = agg.get("module_mean_scores") or {}
    if not means:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(means.keys())
    values = list(means.values())
    bars = ax.bar(labels, values, color="#22c55e")
    ax.set_ylim(0, 1.0)
    ax.set_title("Module average risk score (lower = quieter module)")
    ax.set_ylabel("Mean risk_score")
    for b, v in zip(bars, values):
        ax.annotate(f"{v:.3f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p = charts_dir / "per_module_risk.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _render_latency_histogram(agg, charts_dir: Path) -> Optional[Path]:
    import matplotlib.pyplot as plt
    import pandas as pd
    df = agg.get("raw_df")
    if df is None or df.empty or "gateway_latency_ms" not in df.columns:
        return None
    lat = pd.to_numeric(df["gateway_latency_ms"], errors="coerce").dropna()
    if lat.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lat, bins=min(30, max(5, int(len(lat) ** 0.5))), color="#8b5cf6", edgecolor="white")
    ax.axvline(agg["p50_latency_ms"], linestyle="--", color="#22c55e",
               label=f"p50 = {agg['p50_latency_ms']:.0f} ms")
    ax.axvline(agg["p95_latency_ms"], linestyle="--", color="#ef4444",
               label=f"p95 = {agg['p95_latency_ms']:.0f} ms")
    ax.set_xlabel("Gateway latency (ms)")
    ax.set_ylabel("Calls")
    ax.set_title("Gateway latency distribution")
    ax.legend()
    fig.tight_layout()
    p = charts_dir / "latency_histogram.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _render_decision_breakdown(agg, charts_dir: Path) -> Optional[Path]:
    import matplotlib.pyplot as plt
    block = agg.get("block_count", 0)
    sanitize = agg.get("sanitize_count", 0)
    allow = agg.get("allow_count", 0)
    if block + sanitize + allow == 0:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["allow", "sanitize", "block"]
    sizes = [allow, sanitize, block]
    colors = ["#22c55e", "#eab308", "#ef4444"]
    ax.pie(sizes, labels=labels, colors=colors, autopct=lambda p: f"{p:.0f}%" if p > 0 else "",
           startangle=90, counterclock=False)
    ax.set_title("Final decision distribution")
    fig.tight_layout()
    p = charts_dir / "decision_breakdown.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def _render_confusion_matrix(agg, charts_dir: Path) -> Optional[Path]:
    import matplotlib.pyplot as plt
    tp, fn, fp, tn = agg.get("tp", 0), agg.get("fn", 0), agg.get("fp", 0), agg.get("tn", 0)
    if (tp + fn + fp + tn) == 0:
        return None
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = [[tp, fn], [fp, tn]]
    im = ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            v = matrix[i][j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > max(map(max, matrix)) / 2 else "black",
                    fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["got=block", "got≠block"])
    ax.set_yticklabels(["expected=block", "expected≠block"])
    ax.set_title("Confusion matrix (block = positive class)")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    p = charts_dir / "confusion_block.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Per-suite breakdown table
# ---------------------------------------------------------------------------
def _per_suite_breakdown(agg) -> List[Dict[str, Any]]:
    import pandas as pd
    df = agg.get("raw_df")
    if df is None or df.empty or "suite" not in df.columns:
        return []
    rows = []
    for suite, sub in df.groupby("suite"):
        n = len(sub)
        if n == 0:
            continue
        dec = sub.get("gateway_decision", pd.Series(dtype=str)).astype(str).str.lower()
        exp = sub.get("expected_decision", pd.Series(dtype=str)).astype(str).str.lower()
        miss = int(pd.to_numeric(sub.get("gateway_miss", pd.Series(dtype=int)),
                                 errors="coerce").fillna(0).sum())
        tp = int(((exp == "block") & (dec == "block")).sum())
        fn = int(((exp == "block") & (dec != "block")).sum())
        fp = int(((exp != "block") & (dec == "block")).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        rows.append({
            "suite": suite, "n": n, "miss": miss,
            "precision": precision, "recall": recall, "f1": f1,
        })
    return sorted(rows, key=lambda r: -r["n"])


# ---------------------------------------------------------------------------
# Tool execution summary (only when tools_local rows present)
# ---------------------------------------------------------------------------
def _tools_summary(agg) -> Optional[Dict[str, Any]]:
    df = agg.get("raw_df")
    if df is None or df.empty or "tool_executed" not in df.columns:
        return None
    import pandas as pd
    te = pd.to_numeric(df["tool_executed"], errors="coerce").fillna(0).astype(int)
    if te.sum() == 0:
        return None
    executed = int(te.sum())
    # Restrict error counting + latency averaging to rows where the tool
    # actually ran. Otherwise legacy (pre-Hafta-14) rows count as errors
    # because `astype(str)` turns NaN into the literal "nan" string.
    executed_rows = df[te == 1]
    err_col = executed_rows.get("tool_error", pd.Series(dtype=str))
    err_str = err_col.fillna("").astype(str).str.strip()
    err_str = err_str.replace({"nan": "", "None": ""})
    errors = int((err_str != "").sum())
    tlat = pd.to_numeric(
        executed_rows.get("tool_latency_ms", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    return {
        "executed": executed,
        "with_error": errors,
        "avg_tool_latency_ms": float(tlat.mean()) if not tlat.empty else 0.0,
    }


# ---------------------------------------------------------------------------
# Markdown emit
# ---------------------------------------------------------------------------
_TITLE = "Unified AI Security Gateway — Evaluation Report"


def _md_table(headers: List[str], rows: List[List[Any]]) -> str:
    sep = "|".join(["---"] * len(headers))
    out = ["|" + "|".join(headers) + "|", "|" + sep + "|"]
    for row in rows:
        out.append("|" + "|".join(str(x) for x in row) + "|")
    return "\n".join(out)


def _emit_markdown(
    agg: Dict[str, Any],
    runs: List[Dict[str, Any]],
    charts: Dict[str, Optional[Path]],
    out_md: Path,
) -> None:
    score = composite_security_score(agg)
    recos = generate_recommendations(agg)
    per_suite = _per_suite_breakdown(agg)
    tools = _tools_summary(agg)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rel = lambda p: f"./{p.name}" if p else ""
    charts_dir_rel = out_md.stem + ".charts"

    lines: List[str] = []
    lines.append(f"# {_TITLE}")
    lines.append("")
    lines.append(f"_Generated: {timestamp}_  ·  Window: last {agg['n_recent_runs']} runs · {agg['n_rows']} requests aggregated")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Executive summary
    lines.append("## Executive summary")
    lines.append("")
    risk_level = ("LOW" if score >= 85 else "MEDIUM" if score >= 70
                  else "HIGH" if score >= 50 else "CRITICAL")
    lines.append(f"- **Security score**: **{score:.1f} / 100** ({risk_level} risk)")
    lines.append(f"- **Coverage**: {agg['n_rows']} requests across {agg['n_recent_runs']} runs")
    lines.append(f"- **Block rate**: {agg['block_rate'] * 100:.1f}%  ·  "
                 f"**Miss rate**: {agg['miss_rate'] * 100:.1f}%")
    lines.append(f"- **Precision / Recall / F1**: "
                 f"{agg['precision']:.3f} / {agg['recall']:.3f} / {agg['f1']:.3f}")
    lines.append(f"- **Latency**: avg {agg['avg_latency_ms']:.0f} ms · "
                 f"p50 {agg['p50_latency_ms']:.0f} ms · "
                 f"p95 {agg['p95_latency_ms']:.0f} ms")
    lines.append("")

    # Score formula transparency
    lines.append("> **Score formula**:  "
                 "`100 × (1 − miss_rate) × (1 − latency_breach_rate) × (precision × recall)`. "
                 "Latency-breach rate is calls > 2σ above mean (proxy until per-module budgets land on the snapshot endpoint).")
    lines.append("")

    if charts.get("decision_breakdown"):
        lines.append(f"![Decision breakdown]({charts_dir_rel}/{charts['decision_breakdown'].name})")
        lines.append("")

    # Per-suite breakdown
    lines.append("## Per-suite breakdown")
    lines.append("")
    if per_suite:
        rows = [[
            r["suite"], r["n"], r["miss"],
            f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1']:.3f}",
        ] for r in per_suite]
        lines.append(_md_table(["suite", "n", "miss", "precision", "recall", "F1"], rows))
    else:
        lines.append("_No per-suite data — registry is empty or rows lack a `suite` column._")
    lines.append("")
    if charts.get("attack_distribution"):
        lines.append(f"![Attack distribution]({charts_dir_rel}/{charts['attack_distribution'].name})")
        lines.append("")

    # Module mean risk
    lines.append("## Per-module average risk")
    lines.append("")
    means = agg.get("module_mean_scores") or {}
    if means:
        rows = [[m, f"{v:.3f}"] for m, v in means.items()]
        lines.append(_md_table(["module", "mean risk_score"], rows))
    else:
        lines.append("_No per-module score columns in the data._")
    lines.append("")
    if charts.get("per_module_risk"):
        lines.append(f"![Per-module risk]({charts_dir_rel}/{charts['per_module_risk'].name})")
        lines.append("")

    # Latency
    lines.append("## Latency distribution")
    lines.append("")
    lines.append(f"- avg: {agg['avg_latency_ms']:.0f} ms")
    lines.append(f"- p50: {agg['p50_latency_ms']:.0f} ms")
    lines.append(f"- p95: {agg['p95_latency_ms']:.0f} ms")
    lines.append(f"- breach proxy (calls > 2σ): {agg['latency_breach_rate'] * 100:.1f}%")
    lines.append("")
    if charts.get("latency_histogram"):
        lines.append(f"![Latency histogram]({charts_dir_rel}/{charts['latency_histogram'].name})")
        lines.append("")

    # Confusion matrix
    lines.append("## Confusion matrix (block class)")
    lines.append("")
    rows = [
        ["expected=block", agg["tp"], agg["fn"]],
        ["expected≠block", agg["fp"], agg["tn"]],
    ]
    lines.append(_md_table(["", "got=block", "got≠block"], rows))
    lines.append("")
    if charts.get("confusion_block"):
        lines.append(f"![Confusion matrix]({charts_dir_rel}/{charts['confusion_block'].name})")
        lines.append("")

    # Tools (only when present)
    if tools:
        lines.append("## Tool execution summary")
        lines.append("")
        lines.append(f"- Real tool invocations: **{tools['executed']}**")
        lines.append(f"- Errors (tool-side): **{tools['with_error']}**")
        lines.append(f"- Avg tool latency: {tools['avg_tool_latency_ms']:.0f} ms")
        lines.append("")
        lines.append("Saldırı vektörleri gateway pre-execution'da durdurulduğu için "
                     "`tool_executed=0` satırlarına denk gelir; meşru istekler real upstream'e iletildiği "
                     "için `tool_executed=1` ve `tool_response_preview` doludur.")
        lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    if recos:
        for r in recos:
            sev = r.get("severity", "info")
            icon = {"critical": "🚨", "warn": "⚠️", "info": "ℹ️"}.get(sev, "•")
            lines.append(f"- {icon} **{sev.upper()}**: {r.get('text', '')}")
    else:
        lines.append("_No actionable findings — system metrics look healthy._")
    lines.append("")

    # Appendix: included runs
    lines.append("## Appendix: included runs")
    lines.append("")
    if runs:
        rows = []
        for r in runs:
            rows.append([
                (r.get("ended_at") or r.get("started_at") or "")[:19].replace("T", " "),
                r.get("run_id", ""),
                r.get("target_id", ""),
                r.get("suite", ""),
                r.get("n_rows") or r.get("n_cases") or 0,
                r.get("exit_code", ""),
            ])
        lines.append(_md_table(
            ["ended_at", "run_id", "target", "suite", "n_rows", "exit_code"], rows
        ))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("_Report generated by `scripts/generate_thesis_report.py`._")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20,
                    help="Maximum recent runs to aggregate (default: 20).")
    ap.add_argument("--out", default="reports/thesis_evaluation.md",
                    help="Output markdown path (charts saved alongside in <stem>.charts/).")
    ap.add_argument("--no-charts", action="store_true",
                    help="Skip chart rendering (faster, useful for text-only smoke tests).")
    args = ap.parse_args()

    _set_headless_backend()

    runs = _load_runs(args.n)
    if not runs:
        print(f"[generate_thesis_report] no runs in {_RUNS_DIR / '_registry.jsonl'}",
              file=sys.stderr)
        return 1

    agg = _aggregate(runs)
    if agg["n_rows"] == 0:
        print("[generate_thesis_report] runs registered but per-run results.csv files are empty",
              file=sys.stderr)
        return 2

    out_md = (_PROJECT_ROOT / args.out).resolve()
    charts_dir = out_md.with_suffix("").parent / (out_md.stem + ".charts")

    charts: Dict[str, Optional[Path]] = {
        "attack_distribution": None,
        "per_module_risk": None,
        "latency_histogram": None,
        "decision_breakdown": None,
        "confusion_block": None,
    }
    if not args.no_charts:
        charts_dir.mkdir(parents=True, exist_ok=True)
        charts["attack_distribution"] = _render_attack_distribution(agg, charts_dir)
        charts["per_module_risk"] = _render_per_module_risk(agg, charts_dir)
        charts["latency_histogram"] = _render_latency_histogram(agg, charts_dir)
        charts["decision_breakdown"] = _render_decision_breakdown(agg, charts_dir)
        charts["confusion_block"] = _render_confusion_matrix(agg, charts_dir)

    _emit_markdown(agg, runs, charts, out_md)

    rendered = [k for k, v in charts.items() if v is not None]
    print(f"[generate_thesis_report] wrote {out_md}")
    print(f"[generate_thesis_report] charts: {len(rendered)} rendered "
          f"({', '.join(rendered) if rendered else 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
