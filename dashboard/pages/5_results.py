"""Results — attack matrix, P / R / F1, FP / FN, latency distribution,
per-module performance. Filterable per attack family or to failed cases.
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


_RUNS = _PROJECT_ROOT / "runs"


st.set_page_config(page_title="Results", page_icon=":bar_chart:", layout="wide")
st.title("Results")

DATASETS = {
    # ── batch / eval artefacts (produced by evaluation/ scripts) ──────────
    "Gateway attack suite (gateway_attack_results.csv)":  "gateway_attack_results.csv",
    "External eval (external_eval_results.csv)":           "external_eval_results.csv",
    "Output guard eval (output_eval_metrics.csv)":         "output_eval_metrics.csv",
    "RAG eval final (rag_eval_final.csv)":                 "rag_eval_final.csv",
    "RAG hybrid raw (rag_advanced_hybrid_metrics.csv)":    "rag_advanced_hybrid_metrics.csv",
    "Baseline comparison (baseline_comparison.csv)":       "baseline_comparison.csv",
    # ── production / live-request telemetry CSVs ──────────────────────────
    "Output guard live (output_security_metrics.csv)":     "output_security_metrics.csv",
    "RAG guard live (rag_final_metrics.csv)":              "rag_final_metrics.csv",
}

choice = st.selectbox("Dataset", list(DATASETS.keys()))
csv_path = _RUNS / DATASETS[choice]
if not csv_path.exists():
    st.warning(f"Missing artefact: `{csv_path.relative_to(_PROJECT_ROOT)}`")
    st.stop()

df = pd.read_csv(csv_path)
st.caption(f"{len(df)} row(s) · `{csv_path.relative_to(_PROJECT_ROOT)}`")


# ---------------------------------------------------------------------------
# Schema helpers — adapt to whichever CSV the user picked.
# ---------------------------------------------------------------------------
def _class_col(df) -> Optional[str]:
    # `attack_type` is emitted by run_attack_suite.py (gateway_attack_results.csv)
    # and run_baseline_comparison.py (baseline_comparison.csv); other producers
    # use `attack_class`, `category`, etc. Order matters — first match wins.
    for c in ("attack_class", "attack_type", "category", "suite",
              "poison_type", "poison_technique"):
        if c in df.columns:
            return c
    return None


def _decision_col(df) -> Optional[str]:
    for c in ("decision", "gateway_decision", "final_decision"):
        if c in df.columns:
            return c
    return None


def _expected_col(df) -> Optional[str]:
    for c in ("expected_decision", "expected"):
        if c in df.columns:
            return c
    return None


def _latency_col(df) -> Optional[str]:
    for c in ("latency_ms", "gateway_latency_ms", "adapter_latency_ms"):
        if c in df.columns:
            return c
    return None


def _module_score_cols(df):
    return [
        c for c in (
            "prompt_score", "rag_score", "agency_score", "output_score",
            "gateway_prompt_score", "gateway_rag_score", "gateway_agency_score",
            "embedding_score", "judge_score",
        ) if c in df.columns
    ]


def _is_failed_row(row, dec_col: str, exp_col: Optional[str]) -> bool:
    if exp_col and dec_col:
        exp = str(row.get(exp_col) or "").lower()
        got = str(row.get(dec_col) or "").lower()
        if exp and got and exp != got:
            return True
    if "status" in row.index:
        if str(row.get("status") or "").upper() in {"FN", "FP", "FAIL", "FAILED"}:
            return True
    if "match_expected" in row.index:
        try:
            return int(row["match_expected"]) == 0
        except (ValueError, TypeError):
            pass
    return False


# ---------------------------------------------------------------------------
# Filters — preset (one toggle per attack suite + a "failed only" shortcut)
#          + generic per-column multiselects
# ---------------------------------------------------------------------------
pc1, pc2, pc3, pc4, pc5 = st.columns(5)
preset_prompt = pc1.checkbox("Prompt only", value=False)
preset_rag = pc2.checkbox("RAG only", value=False)
preset_agency = pc3.checkbox("Agency only", value=False)
preset_failed = pc4.checkbox("Failed only", value=False)
if pc5.button("Clear", width="stretch"):
    preset_rag = preset_prompt = preset_agency = preset_failed = False

filters = {}
for col in ("attack_class", "attack_type", "category", "decision",
            "expected_decision", "status", "config", "strategy"):
    if col in df.columns:
        opts = sorted(df[col].dropna().astype(str).unique().tolist())
        if 1 < len(opts) <= 30:
            picked = st.multiselect(col, opts, default=opts)
            filters[col] = picked

filtered = df.copy()
for col, picked in filters.items():
    filtered = filtered[filtered[col].astype(str).isin(picked)]

class_col = _class_col(filtered)
dec_col = _decision_col(filtered)
exp_col = _expected_col(filtered)

if preset_rag and class_col:
    filtered = filtered[filtered[class_col].astype(str).str.contains(
        "rag|poison|chunk", case=False, na=False)]
elif preset_rag:
    st.caption("RAG preset skipped — no class column in this dataset.")
if preset_prompt and class_col:
    filtered = filtered[filtered[class_col].astype(str).str.contains(
        "prompt|inject|jailbreak", case=False, na=False)]
elif preset_prompt:
    st.caption("Prompt preset skipped — no class column in this dataset.")
if preset_agency and class_col:
    # Matches `agency_*` rows from gateway_attack_results.csv /
    # baseline_comparison.csv plus generic `tool` / `idor` / `enumeration`
    # category names used by other agency datasets.
    filtered = filtered[filtered[class_col].astype(str).str.contains(
        "agency|tool|idor|enumeration|role_misuse|cross_user|unauthorized",
        case=False, na=False)]
elif preset_agency:
    st.caption("Agency preset skipped — no class column in this dataset.")
if preset_failed and dec_col:
    failed_mask = filtered.apply(_is_failed_row, axis=1, dec_col=dec_col, exp_col=exp_col)
    filtered = filtered[failed_mask]


# ---------------------------------------------------------------------------
# 1) Attack matrix — class × decision cross-tab
# ---------------------------------------------------------------------------
st.subheader("Attack matrix")
if class_col and dec_col and not filtered.empty:
    matrix = pd.crosstab(filtered[class_col], filtered[dec_col]).sort_index()
    matrix["TOTAL"] = matrix.sum(axis=1)
    st.dataframe(matrix, width="stretch")
else:
    st.info(
        "Need both a class column "
        "(`attack_class` / `attack_type` / `category` / `suite` / `poison_*`) "
        "and a decision column to render the matrix."
    )


# ---------------------------------------------------------------------------
# 2) Confusion matrix + Precision / Recall / F1 + FP / FN counts
#    (block treated as the positive class)
# ---------------------------------------------------------------------------
st.subheader("Precision / Recall / F1 (block = positive)")
if exp_col and dec_col and not filtered.empty:
    sub = filtered.dropna(subset=[exp_col, dec_col]).copy()
    sub[exp_col] = sub[exp_col].astype(str).str.lower()
    sub[dec_col] = sub[dec_col].astype(str).str.lower()

    tp = int(((sub[exp_col] == "block") & (sub[dec_col] == "block")).sum())
    fn = int(((sub[exp_col] == "block") & (sub[dec_col] != "block")).sum())
    fp = int(((sub[exp_col] != "block") & (sub[dec_col] == "block")).sum())
    tn = int(((sub[exp_col] != "block") & (sub[dec_col] != "block")).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    cm = pd.DataFrame(
        [[tp, fn], [fp, tn]],
        index=["expected=block", "expected≠block"],
        columns=["got=block", "got≠block"],
    )
    cA, cB = st.columns([1, 2])
    cA.dataframe(cm, width="stretch")
    cB.metric("Precision", f"{precision:.3f}")
    cB.metric("Recall (TPR)", f"{recall:.3f}")
    cB.metric("F1", f"{f1:.3f}")

    fc1, fc2 = st.columns(2)
    fc1.metric("False Positives", fp, help="expected≠block but gateway said block")
    fc2.metric("False Negatives", fn, help="expected=block but gateway didn't")

    if fp + fn > 0:
        with st.expander(f"FP / FN sample rows (up to 10 of each)", expanded=False):
            st.markdown("**False Positives**")
            st.dataframe(
                sub[(sub[exp_col] != "block") & (sub[dec_col] == "block")].head(10),
                width="stretch", hide_index=True,
            )
            st.markdown("**False Negatives**")
            st.dataframe(
                sub[(sub[exp_col] == "block") & (sub[dec_col] != "block")].head(10),
                width="stretch", hide_index=True,
            )
else:
    st.info(
        "No `expected_decision` / `decision` columns in this dataset — "
        "P/R/F1 only meaningful for evaluation runs with ground truth."
    )


# ---------------------------------------------------------------------------
# 3) Latency distribution
# ---------------------------------------------------------------------------
st.subheader("Latency distribution")
lat_col = _latency_col(filtered)
if lat_col and not filtered.empty:
    lat = pd.to_numeric(filtered[lat_col], errors="coerce").dropna()
    if not lat.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("count", int(len(lat)))
        c2.metric("mean", f"{lat.mean():.0f} ms")
        c3.metric("p50", f"{lat.quantile(0.5):.0f} ms")
        c4.metric("p95", f"{lat.quantile(0.95):.0f} ms")
        # Streamlit's bar_chart requires bucketed data — pandas histogram does that.
        bins = min(20, max(5, int(len(lat) ** 0.5)))
        hist = pd.cut(lat, bins=bins).value_counts().sort_index()
        hist.index = [f"{int(iv.left)}-{int(iv.right)}" for iv in hist.index]
        st.bar_chart(hist)
    else:
        st.info(f"`{lat_col}` column has no numeric values.")
else:
    st.info("No latency column found in this dataset.")


# ---------------------------------------------------------------------------
# 4) Per-module performance — score columns when present
# ---------------------------------------------------------------------------
st.subheader("Per-module performance")
score_cols = _module_score_cols(filtered)
if score_cols and not filtered.empty:
    stats = (
        filtered[score_cols]
        .apply(pd.to_numeric, errors="coerce")
        .agg(["count", "mean", "median", "max"])
        .T.rename(columns={"count": "n", "mean": "mean", "median": "p50", "max": "max"})
    )
    st.dataframe(stats, width="stretch")
    st.caption("Distribution per module (use the chart below to compare).")
    st.bar_chart(filtered[score_cols].apply(pd.to_numeric, errors="coerce").mean())
else:
    st.info("No per-module score columns in this dataset.")


# ---------------------------------------------------------------------------
# 5) Filtered rows — last so users can scroll past the analytics
# ---------------------------------------------------------------------------
st.subheader("Filtered rows")
st.dataframe(filtered, width="stretch", hide_index=True)
