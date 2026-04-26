"""reporting/report_generator.py
End-to-end orchestrator: read on-disk artefacts, compose a thesis-style
markdown report, and write it to `reports/chatbot_security_report.md`.

The function is split into:
    * `render_report(snapshot, alerts, breakers, modules, recommendations)`
        — pure composition, used by tests
    * `generate_report(output_path=None, event_limit=2000)`
        — top-level wrapper that does the I/O

We keep both reachable so unit tests don't need a populated telemetry file.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from monitoring.alert_rules import build_snapshot_from_events, evaluate
from reporting.recommendation_engine import derive_recommendations, render_recommendations
from reporting.summary_generator import build_summary, render_summary


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _PROJECT_ROOT / "reports" / "chatbot_security_report.md"
_RUNS_DIR = _PROJECT_ROOT / "runs"

# Sources for escaped-attack and explainability sections. Optional — the
# report renders gracefully if any are missing (clean install / first run).
_ESCAPED_ATTACK_SOURCES = (
    _RUNS_DIR / "gateway_attack_results.csv",
    _RUNS_DIR / "external_eval_results.csv",
)
# Batch-eval explainability files (written by evaluation/ scripts).
# Production live-telemetry CSVs (output_explainability_log.csv /
# rag_explainability_log.csv) use a different schema and must not collide.
_OUTPUT_EXPLAIN_CSV = _RUNS_DIR / "output_eval_explain.csv"
_RAG_EXPLAIN_CSV = _RUNS_DIR / "rag_eval_explain.csv"


# ---------------------------------------------------------------------------
# Pure rendering — easy to unit-test
# ---------------------------------------------------------------------------
def _render_alerts(alerts: List[Dict[str, Any]]) -> str:
    if not alerts:
        return "## Active alerts\n\n_None._\n"
    lines = ["## Active alerts", ""]
    lines.append("| Severity | Rule | Message |")
    lines.append("|---|---|---|")
    for a in alerts:
        sev = a.get("severity", "?")
        rid = a.get("rule_id", "?")
        msg = (a.get("message") or "").replace("|", "\\|")
        lines.append(f"| {sev} | `{rid}` | {msg} |")
    lines.append("")
    return "\n".join(lines)


def _render_breakers(breakers: List[Dict[str, Any]]) -> str:
    if not breakers:
        return "## Circuit breakers\n\n_No registered breakers._\n"
    lines = ["## Circuit breakers", ""]
    lines.append("| Name | State | Cons. fails | Total fails | Short-circuits |")
    lines.append("|---|---|---:|---:|---:|")
    for b in breakers:
        lines.append(
            f"| `{b.get('name','?')}` | {b.get('state','?')} | "
            f"{b.get('consecutive_failures', 0)} | "
            f"{b.get('total_failures', 0)} | "
            f"{b.get('total_short_circuits', 0)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_module_table(snapshot: Dict[str, float]) -> str:
    """One row per module with avg latency + error rate pulled from snapshot."""
    rows: List[Dict[str, Any]] = []
    for k in snapshot:
        if k.startswith("module_") and k.endswith("_avg_latency_ms"):
            name = k[len("module_"): -len("_avg_latency_ms")]
            rows.append({
                "name": name,
                "avg_latency_ms": float(snapshot.get(k, 0.0) or 0.0),
                "error_rate": float(
                    snapshot.get(f"module_{name}_error_rate", 0.0) or 0.0
                ),
            })
    if not rows:
        return "## Per-module performance\n\n_No module results in window._\n"
    rows.sort(key=lambda r: r["name"])
    lines = ["## Per-module performance", ""]
    lines.append("| Module | Avg latency (ms) | Error rate |")
    lines.append("|---|---:|---:|")
    for r in rows:
        lines.append(
            f"| `{r['name']}` | {r['avg_latency_ms']:.0f} | "
            f"{r['error_rate'] * 100:.2f}% |"
        )
    lines.append("")
    return "\n".join(lines)


def _read_csv_rows(path: Path, *, limit: int = 1000) -> List[Dict[str, str]]:
    """Read up to `limit` rows of a CSV file, returning [] on any failure."""
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [row for _, row in zip(range(limit), reader)]
    except (OSError, csv.Error):
        return []


def _render_escaped_attacks(*, top_n: int = 10) -> str:
    """List concrete cases where the gateway let an attack through.

    Aligns with the `gateway_miss` definition used by
    `external_eval/run_external_eval.py::_classify_gateway_miss`:

        escape = expected ∈ {block, sanitize} AND got = allow

    Cases where the gateway *downgraded* the response (e.g. expected=block,
    got=sanitize) are not escapes — protection still fired — but they are
    rendered in a separate "downgraded" sub-table so the user can audit
    threshold drift between design intent and gateway behaviour.

    Pulls from gateway_attack_results.csv and external_eval_results.csv.
    """
    # Lazy import keeps this module free of an external_eval dependency
    # at top-level (the runner imports report_generator in some flows).
    from external_eval.run_external_eval import _classify_gateway_miss  # noqa: WPS433

    escaped: List[Dict[str, str]] = []
    downgraded: List[Dict[str, str]] = []
    source_used: Optional[str] = None
    for src in _ESCAPED_ATTACK_SOURCES:
        candidate = _read_csv_rows(src)
        if not candidate:
            continue
        source_used = src.name
        for r in candidate:
            expected = (r.get("expected_decision") or "").strip().lower()
            decision = (
                r.get("decision")
                or r.get("gateway_decision")
                or r.get("final_decision")
                or ""
            ).strip().lower()
            if not expected or not decision:
                continue
            row = {
                "attack_class": r.get("attack_class") or r.get("attack_type")
                                or r.get("category") or r.get("suite") or "?",
                "id": r.get("case_id") or r.get("id") or r.get("doc_id") or "?",
                "preview": (r.get("prompt") or r.get("attack_prompt") or r.get("text") or "")[:80],
                "expected": expected,
                "got": decision,
            }
            # Canonical miss check — same predicate as external_eval.
            if _classify_gateway_miss(expected, decision) == 1:
                escaped.append(row)
            elif expected == "block" and decision in {"sanitize", "flag"}:
                downgraded.append(row)
        if escaped or downgraded:
            break

    lines = ["## Escaped attacks", ""]
    lines.append(
        "_Escape definition aligned with `external_eval` `gateway_miss`: "
        "`expected ∈ {block, sanitize}` and `got = allow`. Rows where the "
        "gateway downgraded a `block`-expected case to `sanitize`/`flag` "
        "are listed separately — protection still fired._"
    )
    lines.append("")

    if not escaped and not downgraded:
        lines.append(
            f"_No escaped or downgraded attacks detected "
            f"(source: {source_used or 'no CSV'})._"
        )
        lines.append("")
        return "\n".join(lines)

    if escaped:
        lines.append(
            f"### Escaped (gateway_miss) — {len(escaped)} case(s) "
            f"(source: `{source_used}`, top {min(top_n, len(escaped))} shown)"
        )
        lines.append("")
        lines.append("| # | attack_class | id | got | expected | prompt preview |")
        lines.append("|---:|---|---|---|---|---|")
        for i, r in enumerate(escaped[:top_n], 1):
            preview = (r["preview"] or "").replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {i} | `{r['attack_class']}` | `{r['id']}` | "
                f"{r['got']} | {r['expected']} | {preview} |"
            )
        lines.append("")
    else:
        lines.append("_No `allow` outcomes for `block`/`sanitize`-expected cases — gateway held the line._")
        lines.append("")

    if downgraded:
        lines.append(
            f"### Downgraded (block → sanitize / flag) — {len(downgraded)} case(s) "
            f"(top {min(top_n, len(downgraded))} shown, not counted as misses)"
        )
        lines.append("")
        lines.append("| # | attack_class | id | got | expected | prompt preview |")
        lines.append("|---:|---|---|---|---|---|")
        for i, r in enumerate(downgraded[:top_n], 1):
            preview = (r["preview"] or "").replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {i} | `{r['attack_class']}` | `{r['id']}` | "
                f"{r['got']} | {r['expected']} | {preview} |"
            )
        lines.append("")

    return "\n".join(lines)


def _render_explainability(*, top_n: int = 5) -> str:
    """Surface per-decision evidence so the report stands alone.

    Output guard: top fired flags + one example evidence per flag.
    RAG guard:    highest-score chunks across poisoned docs.

    Source files (batch-eval, NOT the production live logs):
        runs/output_eval_explain.csv  ← evaluation/run_output_guard_batch.py
        runs/rag_eval_explain.csv     ← evaluation/build_rag_artefacts.py

    The live per-request logs (`output_explainability_log.csv` /
    `rag_explainability_log.csv`) use a different schema and are
    consumed by the Streamlit Logs page, not by this report.
    """
    lines = ["## Explainability", ""]

    # Output guard — group by `flag`, keep one example evidence each
    out_rows = _read_csv_rows(_OUTPUT_EXPLAIN_CSV)
    if out_rows:
        per_flag: Dict[str, Dict[str, Any]] = {}
        for r in out_rows:
            flag = (r.get("flag") or "?").strip()
            slot = per_flag.setdefault(flag, {"count": 0, "evidence": "", "example_id": ""})
            slot["count"] += 1
            if not slot["evidence"]:
                slot["evidence"] = (r.get("evidence") or "")[:120]
                slot["example_id"] = r.get("id") or ""
        ranked = sorted(per_flag.items(), key=lambda kv: -kv[1]["count"])[:top_n]
        lines.append("### Output guard — flags fired in batch run")
        lines.append("")
        lines.append("| Flag | Count | Example id | Evidence (truncated) |")
        lines.append("|---|---:|---|---|")
        for flag, info in ranked:
            ev = (info["evidence"] or "").replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| `{flag}` | {info['count']} | "
                f"`{info['example_id']}` | {ev} |"
            )
        lines.append("")
    else:
        lines.append(
            f"_No `{_OUTPUT_EXPLAIN_CSV.relative_to(_PROJECT_ROOT)}` — "
            "run `python evaluation/run_output_guard_batch.py` to populate it. "
            "(The live per-request log "
            "`runs/output_explainability_log.csv` uses a different schema "
            "and is rendered by the dashboard Logs page, not this report.)_"
        )
        lines.append("")

    # RAG guard — top chunks by judge_score, only those that drove a non-allow
    rag_rows = _read_csv_rows(_RAG_EXPLAIN_CSV)
    if rag_rows:
        scored: List[Dict[str, Any]] = []
        for r in rag_rows:
            try:
                sc = float(r.get("chunk_judge_score") or 0.0)
            except ValueError:
                continue
            decision = (r.get("decision") or "").lower()
            if decision in {"allow", "", None}:
                continue
            scored.append({
                "doc_id": r.get("doc_id"),
                "chunk_idx": r.get("chunk_idx"),
                "score": sc,
                "decision": decision,
                "preview": (r.get("chunk_text_preview") or "")[:100],
            })
        scored.sort(key=lambda r: -r["score"])
        if scored:
            lines.append("### RAG guard — chunks that drove block / sanitize")
            lines.append("")
            lines.append("| doc_id | chunk_idx | judge_score | decision | preview |")
            lines.append("|---|---:|---:|---|---|")
            for r in scored[:top_n]:
                preview = r["preview"].replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| `{r['doc_id']}` | {r['chunk_idx']} | "
                    f"{r['score']:.2f} | {r['decision']} | {preview} |"
                )
            lines.append("")
    else:
        lines.append(
            f"_No `{_RAG_EXPLAIN_CSV.relative_to(_PROJECT_ROOT)}` — "
            "run `python evaluation/build_rag_artefacts.py` to populate it. "
            "(The live per-request log "
            "`runs/rag_explainability_log.csv` uses a different schema "
            "and is rendered by the dashboard Logs page, not this report.)_"
        )
        lines.append("")

    return "\n".join(lines)


def _render_glossary() -> str:
    """One-line definition of every metric mentioned in this report."""
    rows = [
        ("`block_rate`",   "Fraction of requests where the fused decision was `block`."),
        ("`sanitize_rate`","Fraction where the gateway downgraded the response (mask / safe-rewrite)."),
        ("`flag_rate`",    "Fraction labelled `flag` — passed to the model with a warning."),
        ("`allow_rate`",   "Fraction passed through untouched."),
        ("`bypass_rate`",  "Heuristic: any module ≥ 0.60 yet final decision = `allow`. Proxy for missed attacks."),
        ("`avg_latency_ms` / `p95_latency_ms`",
                           "Total fusion-pipeline latency per request (mean and 95th percentile)."),
        ("`error_rate`",   "Fraction of telemetry events with module-level errors."),
        ("`module_<name>_avg_latency_ms`",
                           "Per-module mean wall-clock time, computed only over requests where the module ran."),
        ("`module_<name>_block_count`",
                           "How often a given module's score crossed the block threshold for a request that ended up blocked."),
    ]
    lines = ["### Metric definitions", ""]
    lines.append("| Metric | Definition |")
    lines.append("|---|---|")
    for name, definition in rows:
        lines.append(f"| {name} | {definition} |")
    lines.append("")
    return "\n".join(lines)


def render_report(
    snapshot: Dict[str, float],
    alerts: List[Dict[str, Any]],
    breakers: List[Dict[str, Any]],
    *,
    generated_at: Optional[str] = None,
    event_count: int = 0,
) -> str:
    """Pure composition. Returns a complete markdown string."""
    summary = build_summary(snapshot, alerts=alerts)
    recommendations = derive_recommendations(snapshot)

    ts = generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")

    parts: List[str] = []
    parts.append(f"# Chatbot Security Report")
    parts.append("")
    parts.append(f"_Generated {ts} · {event_count} telemetry events evaluated._")
    parts.append("")
    parts.append(render_summary(summary))
    parts.append(_render_module_table(snapshot))
    parts.append(_render_escaped_attacks())
    parts.append(_render_explainability())
    parts.append(_render_alerts(alerts))
    parts.append(_render_breakers(breakers))
    parts.append(render_recommendations(recommendations))

    parts.append("## Methodology")
    parts.append("")
    parts.append(
        "Metrics are derived from `logs/system_telemetry.jsonl` via "
        "`monitoring.alert_rules.build_snapshot_from_events`. The bypass "
        "proxy uses the heuristic _module_max ≥ 0.60 ∧ fused decision = allow_; "
        "it counts how often the gateway let through a request that at least "
        "one module flagged as elevated. Recommendations are rule-based "
        "(`reporting/recommendation_engine.py`) — they fire on metric "
        "thresholds, not on model output. The full rule list is auditable in "
        "source."
    )
    parts.append("")
    parts.append(_render_glossary())
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# I/O wrapper
# ---------------------------------------------------------------------------
def _collect_breakers() -> List[Dict[str, Any]]:
    """Snapshot the in-process circuit-breaker registry."""
    try:
        from utils.fallback_handler import _REGISTRY as _R  # noqa: SLF001
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for name, br in _R.items():
        try:
            s = br.stats()
            out.append({
                "name": name,
                "state": s.state.value,
                "consecutive_failures": s.consecutive_failures,
                "total_failures": s.total_failures,
                "total_short_circuits": s.total_short_circuits,
            })
        except Exception:
            continue
    return out


def generate_report(
    output_path: Optional[Path] = None,
    event_limit: int = 2000,
    *,
    events: Optional[Iterable[Dict[str, Any]]] = None,
) -> Path:
    """Write a report to disk; return the resolved path.

    `events` argument lets tests inject a synthetic telemetry stream without
    touching the real jsonl log.
    """
    from schemas import telemetry_schema as ts

    output_path = Path(output_path) if output_path else _DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if events is None:
        try:
            evs = ts.read_events(limit=event_limit)
        except FileNotFoundError:
            evs = []
    else:
        evs = list(events)

    snapshot = build_snapshot_from_events(evs)
    alerts = [a.to_dict() for a in evaluate(snapshot)]
    breakers = _collect_breakers()

    body = render_report(
        snapshot,
        alerts,
        breakers,
        event_count=len(list(evs)) if events is not None else len(evs),
    )
    output_path.write_text(body, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate the chatbot security report.")
    ap.add_argument("--output", "-o", default=str(_DEFAULT_OUTPUT))
    ap.add_argument("--event-limit", type=int, default=2000)
    args = ap.parse_args()

    out = generate_report(Path(args.output), event_limit=args.event_limit)
    print(f"[report] wrote {out}")
