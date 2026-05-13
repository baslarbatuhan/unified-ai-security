"""Results — run-first analytics for completed external_eval / suite runs.

Reads `runs/_registry.jsonl` for the run picker, then loads the
selected run's per-run `runs/<run_id>/results.csv` (or falls back to
filtering the legacy aggregate `runs/external_eval_results.csv` when
the manifest layer hasn't seen this run yet).

Sections:
    Overview        — KPI cards (total / block / sanitize / allow / miss / latency)
    Decisions       — final_decision distribution + decision_band split
    Latency         — p50/p95 metrics + histogram
    Module Scores   — prompt / rag / agency / output mean + bars
    Failures        — FN / FP rows the gateway got wrong
    Raw Data        — full filtered table + CSV export

Decision semantics shown in the UI:
    final_decision   → operational outcome (allow / sanitize / block)
    decision_band    → audit detail (allow / sanitize / flag / block)
                       'flag' is suspicion-tier block, see _collapse_band.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from utils.run_manifest import (
    RESULTS_FILENAME,
    read_manifest,
    read_registry,
)
from dashboard.lib.components import (
    decision_card,
    risk_gauge,
    run_selector_widget,
)


_RUNS_DIR = _PROJECT_ROOT / "runs"
_LEGACY_AGG_CSV = _RUNS_DIR / "external_eval_results.csv"


st.set_page_config(page_title="Results", page_icon=":bar_chart:", layout="wide")
st.title("Results")
st.caption("Per-run analytics. Pick a run from the registry; data narrows automatically.")


# ---------------------------------------------------------------------------
# Run picker — reads runs/_registry.jsonl. Newest first, with target+suite
# in the label so users can pick without remembering opaque ids.
# ---------------------------------------------------------------------------
registry = read_registry(_RUNS_DIR)

if not registry:
    st.warning(
        "No runs registered yet. Launch a run from the **Run test** page, "
        "or backfill from the legacy aggregate via "
        "`python scripts/backfill_run_registry.py`."
    )
    st.stop()


# Hafta 15: `run_selector_widget` ships the label + selectbox + URL
# query-param sync that this page used to inline. Same UX, one source
# of truth for label format and deep-link behaviour across pages.
entry = run_selector_widget(registry, key="results_run_picker")
if entry is None:
    st.stop()
run_id = entry.get("run_id") or ""


# ---------------------------------------------------------------------------
# Resolve the data source: per-run results.csv (preferred) → legacy filter.
# ---------------------------------------------------------------------------
run_dir = _RUNS_DIR / run_id
per_run_csv = run_dir / RESULTS_FILENAME

df: pd.DataFrame
source_label: str

if per_run_csv.exists():
    df = pd.read_csv(per_run_csv)
    source_label = f"per-run · `{per_run_csv.relative_to(_PROJECT_ROOT)}`"
elif _LEGACY_AGG_CSV.exists():
    full = pd.read_csv(_LEGACY_AGG_CSV)
    df = full[full["run_id"].astype(str) == run_id].reset_index(drop=True)
    source_label = (
        f"legacy aggregate filtered · `{_LEGACY_AGG_CSV.relative_to(_PROJECT_ROOT)}` "
        "(no per-run results.csv yet — try backfill)"
    )
else:
    df = pd.DataFrame()
    source_label = "no source found"


# ---------------------------------------------------------------------------
# Run summary header — what the user is looking at
# ---------------------------------------------------------------------------
manifest = read_manifest(run_dir)
header_cols = st.columns(5)
header_cols[0].metric("Run id", run_id[-12:] if len(run_id) > 12 else run_id, help=run_id)
header_cols[1].metric("Target", entry.get("target_id") or "?")
header_cols[2].metric("Suite", entry.get("suite") or "?")
header_cols[3].metric("Cases", int(entry.get("n_rows") or entry.get("n_cases") or 0))
exit_code = entry.get("exit_code")
status_str = "✓ done" if exit_code == 0 else (f"✗ exit={exit_code}" if exit_code is not None else "?")
header_cols[4].metric("Status", status_str)

st.caption(f"Source: {source_label}")


# ---------------------------------------------------------------------------
# Guided "missing artefact" panel — replaces the old st.warning short circuit.
# ---------------------------------------------------------------------------
if df.empty:
    st.error("No rows for this run.")
    with st.expander("What can I do?", expanded=True):
        st.markdown(
            f"""
- The run completed but its rows aren't in `external_eval_results.csv` either —
  this happens for runs launched without `--gateway-analyze` or that crashed
  mid-run.
- Check the runner log:
"""
        )
        runner_log = run_dir / "runner.log"
        if runner_log.exists():
            with st.expander(f"runner.log ({runner_log.stat().st_size} bytes)"):
                st.code(runner_log.read_text(encoding="utf-8", errors="replace")[:8000], language="text")
        else:
            st.info("No runner.log found.")
        st.markdown("- Or re-run the suite from the **Run test** page.")
    if manifest:
        with st.expander("manifest.json"):
            st.json(manifest)
    st.stop()


# ---------------------------------------------------------------------------
# Quick filters — applied to all tabs
# ---------------------------------------------------------------------------
fc1, fc2, fc3 = st.columns([2, 2, 2])
failed_only = fc1.checkbox("Failed only (gateway_miss=1)", value=False)
band_filter = fc2.multiselect(
    "decision_band",
    sorted([b for b in df.get("gateway_decision_band", pd.Series(dtype=str)).dropna().unique() if b]),
    default=[],
    help="Empty = all bands. Use this to inspect flag-tier vs confident block.",
)
cat_filter = fc3.multiselect(
    "category",
    sorted([c for c in df.get("category", pd.Series(dtype=str)).dropna().unique() if c]),
    default=[],
    help="Empty = all categories.",
)

filtered = df.copy()
if failed_only and "gateway_miss" in filtered.columns:
    filtered = filtered[pd.to_numeric(filtered["gateway_miss"], errors="coerce") == 1]
if band_filter and "gateway_decision_band" in filtered.columns:
    filtered = filtered[filtered["gateway_decision_band"].astype(str).isin(band_filter)]
if cat_filter and "category" in filtered.columns:
    filtered = filtered[filtered["category"].astype(str).isin(cat_filter)]

st.caption(f"After filters: {len(filtered)} / {len(df)} row(s)")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_dec, tab_lat, tab_mod, tab_fail, tab_perf, tab_raw = st.tabs(
    ["Overview", "Decisions", "Latency", "Module Scores", "Failures",
     "Performance", "Raw Data"]
)


# --- helpers used by multiple tabs -----------------------------------------
def _decision_counts(d: pd.DataFrame, col: str) -> pd.Series:
    if col not in d.columns:
        return pd.Series(dtype=int)
    return d[col].astype(str).replace("", pd.NA).dropna().value_counts()


# --- Overview ---------------------------------------------------------------
with tab_overview:
    total = len(filtered)
    fd = _decision_counts(filtered, "gateway_decision")
    block_n = int(fd.get("block", 0))
    sanitize_n = int(fd.get("sanitize", 0))
    allow_n = int(fd.get("allow", 0))
    miss_n = int(pd.to_numeric(filtered.get("gateway_miss", pd.Series(dtype=int)), errors="coerce").fillna(0).sum()) \
        if "gateway_miss" in filtered.columns else 0
    avg_lat = float(pd.to_numeric(filtered.get("gateway_latency_ms", pd.Series(dtype=float)), errors="coerce").mean() or 0.0)

    o1, o2, o3, o4, o5, o6 = st.columns(6)
    o1.metric("Total", total)
    o2.metric("Block", block_n)
    o3.metric("Sanitize", sanitize_n)
    o4.metric("Allow", allow_n)
    o5.metric("Miss", miss_n, help="gateway_miss=1: expected block/sanitize but gateway returned allow.")
    o6.metric("Avg latency", f"{avg_lat:.0f} ms")

    if "expected_decision" in filtered.columns and "gateway_decision" in filtered.columns:
        sub = filtered.dropna(subset=["expected_decision", "gateway_decision"]).copy()
        sub["expected_decision"] = sub["expected_decision"].astype(str).str.lower()
        sub["gateway_decision"] = sub["gateway_decision"].astype(str).str.lower()
        tp = int(((sub["expected_decision"] == "block") & (sub["gateway_decision"] == "block")).sum())
        fn = int(((sub["expected_decision"] == "block") & (sub["gateway_decision"] != "block")).sum())
        fp = int(((sub["expected_decision"] != "block") & (sub["gateway_decision"] == "block")).sum())
        tn = int(((sub["expected_decision"] != "block") & (sub["gateway_decision"] != "block")).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

        st.subheader("Precision / Recall / F1 (block = positive)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", f"{precision:.3f}")
        m2.metric("Recall", f"{recall:.3f}")
        m3.metric("F1", f"{f1:.3f}")
        cm = pd.DataFrame(
            [[tp, fn], [fp, tn]],
            index=["expected=block", "expected≠block"],
            columns=["got=block", "got≠block"],
        )
        st.dataframe(cm, width="stretch")


# --- Decisions --------------------------------------------------------------
with tab_dec:
    st.subheader("final_decision (operational outcome)")
    fd_counts = _decision_counts(filtered, "gateway_decision")
    if not fd_counts.empty:
        st.bar_chart(fd_counts)
    else:
        st.info("No decision data.")

    st.subheader("decision_band (audit detail)")
    st.caption(
        "`flag` band = suspicion-tier block. Operationally treated as block "
        "(blocks traffic), but tagged differently for analyst triage."
    )
    band_counts = _decision_counts(filtered, "gateway_decision_band")
    if not band_counts.empty:
        st.bar_chart(band_counts)
    else:
        st.info("decision_band column missing — older run, pre-flag-collapse.")


# --- Latency ----------------------------------------------------------------
with tab_lat:
    if "gateway_latency_ms" in filtered.columns:
        lat = pd.to_numeric(filtered["gateway_latency_ms"], errors="coerce").dropna()
        if not lat.empty:
            l1, l2, l3, l4 = st.columns(4)
            l1.metric("Count", int(len(lat)))
            l2.metric("Mean", f"{lat.mean():.0f} ms")
            l3.metric("p50", f"{lat.quantile(0.5):.0f} ms")
            l4.metric("p95", f"{lat.quantile(0.95):.0f} ms")
            bins = min(20, max(5, int(len(lat) ** 0.5)))
            hist = pd.cut(lat, bins=bins).value_counts().sort_index()
            hist.index = [f"{int(iv.left)}-{int(iv.right)}" for iv in hist.index]
            st.bar_chart(hist)
        else:
            st.info("No numeric latency values.")
    else:
        st.info("No `gateway_latency_ms` column.")


# --- Module Scores ----------------------------------------------------------
with tab_mod:
    score_cols = [
        c for c in (
            "gateway_prompt_score",
            "gateway_rag_score",
            "gateway_agency_score",
            "gateway_fused_score",
        ) if c in filtered.columns
    ]
    if score_cols:
        nums = filtered[score_cols].apply(pd.to_numeric, errors="coerce")
        stats = nums.agg(["count", "mean", "median", "max"]).T.rename(
            columns={"count": "n", "mean": "mean", "median": "p50", "max": "max"}
        )
        st.dataframe(stats, width="stretch")
        st.caption("Mean per module — spot which detector carried the run.")
        st.bar_chart(nums.mean())
    else:
        st.info("No per-module score columns in this dataset.")


# --- Failures ---------------------------------------------------------------
with tab_fail:
    if "expected_decision" in filtered.columns and "gateway_decision" in filtered.columns:
        sub = filtered.dropna(subset=["expected_decision", "gateway_decision"]).copy()
        sub["expected_decision"] = sub["expected_decision"].astype(str).str.lower()
        sub["gateway_decision"] = sub["gateway_decision"].astype(str).str.lower()
        fps = sub[(sub["expected_decision"] != "block") & (sub["gateway_decision"] == "block")]
        fns = sub[(sub["expected_decision"] == "block") & (sub["gateway_decision"] != "block")]
        st.subheader(f"False Negatives — gateway missed an attack ({len(fns)})")
        if fns.empty:
            st.success("No false negatives in this run.")
        else:
            st.dataframe(fns, width="stretch", hide_index=True)
        st.subheader(f"False Positives — gateway blocked legitimate input ({len(fps)})")
        if fps.empty:
            st.success("No false positives in this run.")
        else:
            st.dataframe(fps, width="stretch", hide_index=True)
    else:
        st.info("Need `expected_decision` + `gateway_decision` columns to compute FN/FP.")


# --- Performance ------------------------------------------------------------
# Hafta 12.2: chunk routing efficiency. Reads from runs/<run_id>/results.csv
# columns (rag-specific telemetry lives in rag_final_metrics.csv, but the
# per-run filter is the same). Built on top of the aggregate counts; no
# per-chunk join required here.
with tab_perf:
    st.caption(
        "Routing efficiency for RAG pipelines: how many chunks were "
        "evaluated, how many actually went to the LLM judge, and how "
        "much latency the routing saved by short-circuiting low-risk chunks."
    )

    # The per-run results.csv carries the external_eval aggregate. The
    # rag-routing columns live in rag_final_metrics.csv (cross-target).
    # For per-run drill-down we load rag_final_metrics filtered by run_id.
    rag_metrics_path = _RUNS_DIR / "rag_final_metrics.csv"
    if rag_metrics_path.exists():
        try:
            rag_df = pd.read_csv(rag_metrics_path)
        except Exception as exc:  # noqa: BLE001
            rag_df = pd.DataFrame()
            st.warning(f"Could not load rag_final_metrics.csv: {exc}")
        if not rag_df.empty and "run_id" in rag_df.columns:
            scoped = rag_df[rag_df["run_id"].astype(str) == run_id]
        else:
            scoped = pd.DataFrame()
    else:
        scoped = pd.DataFrame()
        st.info("`rag_final_metrics.csv` not found — no routing data yet.")

    if scoped.empty:
        st.info(
            "No rag_guard rows for this run — the suite may not exercise "
            "the RAG pipeline, or the run predates Hafta 12.2 columns."
        )
    else:
        # KPI strip — sums and averages across all rag_guard calls in run.
        def _sum(col: str) -> float:
            if col not in scoped.columns:
                return 0.0
            return float(pd.to_numeric(scoped[col], errors="coerce").fillna(0).sum())

        def _mean(col: str) -> float:
            if col not in scoped.columns:
                return 0.0
            return float(pd.to_numeric(scoped[col], errors="coerce").mean() or 0.0)

        total_chunks = int(_sum("total_chunks_evaluated"))
        total_judge_calls = int(_sum("total_llm_judge_calls"))
        avg_savings = _mean("routing_savings_pct")
        sum_emb = int(_sum("embedding_phase_ms"))
        sum_jud = int(_sum("judge_phase_ms"))
        chunks_saved = max(total_chunks - total_judge_calls, 0)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total chunks", total_chunks)
        c2.metric("Judge calls", total_judge_calls,
                  help="FAST_JUDGE + DEEP_JUDGE. SKIP chunks never hit the LLM.")
        c3.metric("Routing savings", f"{avg_savings:.1f} %",
                  help="Mean of routing_savings_pct across rag_guard calls.")
        c4.metric("LLM calls avoided", chunks_saved)
        c5.metric("Embedding ms (sum)", f"{sum_emb:,}")

        # Latency split bar: embedding phase vs judge phase.
        st.subheader("Latency split (sum across run)")
        split_df = pd.DataFrame(
            {"phase": ["embedding", "judge"], "ms": [sum_emb, sum_jud]}
        ).set_index("phase")
        st.bar_chart(split_df)
        st.caption(
            "Embedding stage is local + cheap; judge stage is LLM-bound. "
            "Routing pushes work from `judge` to `embedding` when the "
            "embedding score is confident in either direction."
        )

        # Route mix as a stacked bar across cases.
        st.subheader("Route mix per case")
        mix_cols = [c for c in ("route_skip", "route_fast_judge", "route_deep_judge")
                    if c in scoped.columns]
        if mix_cols and "case_id" in scoped.columns:
            mix = scoped[["case_id"] + mix_cols].set_index("case_id")
            st.bar_chart(mix)
        else:
            st.caption("No per-case route breakdown in this dataset.")

        with st.expander("Raw rag_final_metrics rows (this run)"):
            st.dataframe(scoped, width="stretch", hide_index=True)


# --- Raw Data ---------------------------------------------------------------
with tab_raw:
    st.dataframe(filtered, width="stretch", hide_index=True)
    st.download_button(
        "Download filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"{run_id}_filtered.csv",
        mime="text/csv",
    )

    # Hafta 12.1: per-case decision trace drill-down. Lets the analyst
    # pick any case_id from the filtered set and see the full fusion
    # chain (module risks → weighted sum → max-rule override → final).
    st.markdown("---")
    st.subheader("🔍 Decision trace (per case)")
    if "case_id" in filtered.columns and not filtered.empty:
        case_choices = filtered["case_id"].astype(str).tolist()
        picked_case = st.selectbox(
            "Pick a case to drill into", case_choices,
            key=f"results_trace_case::{run_id}",
        )
        if picked_case:
            try:
                from dashboard.lib.gateway_client import GatewayError, get_default_client
                gw = get_default_client()
                trace_resp = gw.get_json(
                    f"/decisions/{run_id}/{picked_case}/trace"
                ) or {}
                trace = trace_resp.get("trace") or {}
            except GatewayError as exc:
                trace = {}
                st.warning(
                    f"No live trace from gateway ({exc}). "
                    "The decision_trace.csv may not exist yet for this run."
                )
            if trace:
                # Top-line: final decision + band + triggering module.
                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric("final_decision", trace.get("final_decision", "?"))
                tc2.metric("decision_band", trace.get("decision_band", "?"))
                tc3.metric("fused_risk", trace.get("fused_risk", "?"))
                tc4.metric("triggering", trace.get("triggering_module", "?"))

                # Visual fused-risk gauge — quick read for the audience
                # before they dive into the formula text below.
                try:
                    fused_val = float(trace.get("fused_risk") or 0.0)
                except (TypeError, ValueError):
                    fused_val = 0.0
                risk_gauge(fused_val, label="fused_risk")

                # Fusion formula prose — one-liner the demo audience can read.
                st.code(trace.get("fusion_formula", ""), language="text")
                if trace.get("override_applied") not in (None, "", "none"):
                    st.caption(
                        f"⚡ Max-rule override applied: **{trace['override_applied']}** "
                        "(single-module max risk pulled the fused score up)."
                    )

                # Module-level breakdown — colour-coded cards (Hafta 13
                # decision_card component). Falls back to a plain table
                # if module_risks is empty / malformed.
                st.markdown("**Per-module breakdown**")
                mods = trace.get("module_risks") or []
                if isinstance(mods, list) and mods:
                    for m in mods:
                        if not isinstance(m, dict):
                            continue
                        decision_card(
                            module=str(m.get("module", "?")),
                            risk=float(m.get("risk_score") or 0.0),
                            decision=str(m.get("decision", "")),
                            evidence=[m.get("top_evidence") or ""] if m.get("top_evidence") else None,
                        )
                else:
                    st.caption("No per-module breakdown captured.")
            else:
                st.info("Trace not available for this case.")
    else:
        st.caption("Trace drill-down needs a `case_id` column.")

    if manifest:
        with st.expander("manifest.json"):
            st.json(manifest)
