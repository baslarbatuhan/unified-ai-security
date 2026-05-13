"""Compare runs — side-by-side delta of two completed runs.

Picks two run_ids from `runs/_registry.jsonl` and renders:
    * Header: target / suite / cases / exit code per side
    * KPI deltas (block / sanitize / allow / miss / avg latency)
    * Decision distribution side-by-side (bar chart)
    * decision_band distribution side-by-side (audit detail)
    * Latency distribution side-by-side
    * Module score means side-by-side
    * Per-case diff for cases that appear in BOTH runs

Useful flows:
    * "Did my config tweak help?" — compare a baseline run vs a tuned run
    * "Did this code change regress recall?" — compare last week vs HEAD
    * "Why is this run worse?" — drill into per-case diff to find regressions
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
    read_registry,
)
from dashboard.lib.components import run_selector_widget


_RUNS_DIR = _PROJECT_ROOT / "runs"
_LEGACY_AGG_CSV = _RUNS_DIR / "external_eval_results.csv"


st.set_page_config(page_title="Compare runs", page_icon=":scales:", layout="wide")
st.title("Compare runs")
st.caption("Side-by-side delta of two completed runs.")


registry = read_registry(_RUNS_DIR)
if len(registry) < 2:
    st.warning(
        f"Need at least 2 registered runs to compare; have {len(registry)}. "
        "Launch more runs from the **Run test** page."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Two-pane run picker via the shared `run_selector_widget`. Each side has
# its own URL query-param field (`left`, `right`) and a sensible default
# (newest = right, second-newest = left) so the page is useful on first
# render without typing anything.
# ---------------------------------------------------------------------------
default_left = registry[1]["run_id"] if len(registry) > 1 else registry[0]["run_id"]
default_right = registry[0]["run_id"]

col_l, col_r = st.columns(2)
with col_l:
    left_entry = run_selector_widget(
        registry, key="cmp_left",
        default_run_id=default_left,
        label="Left run (baseline)",
        qp_field="left",
    )
with col_r:
    right_entry = run_selector_widget(
        registry, key="cmp_right",
        default_run_id=default_right,
        label="Right run (compare)",
        qp_field="right",
    )

# `run_selector_widget` returns None only on empty registry; we already
# short-circuited above when len(registry) < 2.
assert left_entry is not None and right_entry is not None
left_id = left_entry.get("run_id") or ""
right_id = right_entry.get("run_id") or ""

if left_id == right_id:
    st.warning("Same run on both sides — pick two different runs to see deltas.")


# ---------------------------------------------------------------------------
# Resolve data per side: prefer per-run results.csv, fall back to legacy.
# ---------------------------------------------------------------------------
def _load(run_id: str) -> pd.DataFrame:
    per = _RUNS_DIR / run_id / RESULTS_FILENAME
    if per.exists():
        return pd.read_csv(per)
    if _LEGACY_AGG_CSV.exists():
        full = pd.read_csv(_LEGACY_AGG_CSV)
        return full[full["run_id"].astype(str) == run_id].reset_index(drop=True)
    return pd.DataFrame()


left_df = _load(left_id)
right_df = _load(right_id)


# ---------------------------------------------------------------------------
# Header — what each side is
# ---------------------------------------------------------------------------
def _header_box(col, side_label: str, entry: dict, df: pd.DataFrame) -> None:
    col.markdown(f"### {side_label}")
    col.caption(entry.get("run_id") or "?")
    cc = col.columns(3)
    cc[0].metric("Target", entry.get("target_id") or "?")
    cc[1].metric("Suite", entry.get("suite") or "?")
    cc[2].metric("Rows", len(df))


hl, hr = st.columns(2)
_header_box(hl, "Left (baseline)", left_entry, left_df)
_header_box(hr, "Right (compare)", right_entry, right_df)


# ---------------------------------------------------------------------------
# KPI delta — block / sanitize / allow / miss / avg latency
# ---------------------------------------------------------------------------
def _kpis(d: pd.DataFrame) -> dict:
    out = {"total": len(d), "block": 0, "sanitize": 0, "allow": 0, "miss": 0, "avg_latency_ms": 0.0}
    if "gateway_decision" in d.columns and not d.empty:
        vc = d["gateway_decision"].astype(str).value_counts()
        out["block"] = int(vc.get("block", 0))
        out["sanitize"] = int(vc.get("sanitize", 0))
        out["allow"] = int(vc.get("allow", 0))
    if "gateway_miss" in d.columns and not d.empty:
        out["miss"] = int(pd.to_numeric(d["gateway_miss"], errors="coerce").fillna(0).sum())
    if "gateway_latency_ms" in d.columns and not d.empty:
        out["avg_latency_ms"] = float(pd.to_numeric(d["gateway_latency_ms"], errors="coerce").mean() or 0.0)
    return out


lk = _kpis(left_df)
rk = _kpis(right_df)

st.subheader("KPI delta (right − left)")
kpi_cols = st.columns(6)
for i, key in enumerate(("total", "block", "sanitize", "allow", "miss", "avg_latency_ms")):
    delta = rk[key] - lk[key]
    label = key.replace("_", " ").title()
    if key == "avg_latency_ms":
        kpi_cols[i].metric(label, f"{rk[key]:.0f} ms", f"{delta:+.0f} ms")
    else:
        # `miss` reduction is GOOD — invert the delta colour by using
        # `delta_color="inverse"`. block can go either direction (more
        # blocks may mean more attacks, not necessarily a regression).
        delta_color = "inverse" if key == "miss" else "normal"
        kpi_cols[i].metric(label, rk[key], f"{delta:+d}", delta_color=delta_color)


# ---------------------------------------------------------------------------
# Distribution comparisons
# ---------------------------------------------------------------------------
def _dist_compare(left: pd.DataFrame, right: pd.DataFrame, col: str) -> Optional[pd.DataFrame]:
    if col not in left.columns and col not in right.columns:
        return None
    lv = left[col].astype(str).value_counts() if col in left.columns else pd.Series(dtype=int)
    rv = right[col].astype(str).value_counts() if col in right.columns else pd.Series(dtype=int)
    keys = sorted(set(lv.index) | set(rv.index))
    out = pd.DataFrame(
        {"left": [int(lv.get(k, 0)) for k in keys], "right": [int(rv.get(k, 0)) for k in keys]},
        index=keys,
    )
    out = out[out.index.astype(str) != ""]
    return out


st.subheader("Decisions")
dec_df = _dist_compare(left_df, right_df, "gateway_decision")
if dec_df is not None and not dec_df.empty:
    st.bar_chart(dec_df)
else:
    st.info("No decision data on either side.")

st.subheader("decision_band (audit)")
band_df = _dist_compare(left_df, right_df, "gateway_decision_band")
if band_df is not None and not band_df.empty:
    st.bar_chart(band_df)
else:
    st.info("Older runs may lack `gateway_decision_band`.")


# ---------------------------------------------------------------------------
# Latency side-by-side
# ---------------------------------------------------------------------------
st.subheader("Latency (gateway_latency_ms)")
def _lat_stats(d: pd.DataFrame) -> dict:
    if "gateway_latency_ms" not in d.columns or d.empty:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
    s = pd.to_numeric(d["gateway_latency_ms"], errors="coerce").dropna()
    if s.empty:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "p50": float(s.quantile(0.5)),
        "p95": float(s.quantile(0.95)),
    }


ls = _lat_stats(left_df)
rs = _lat_stats(right_df)
lat_table = pd.DataFrame(
    {"left": [ls["n"], ls["mean"], ls["p50"], ls["p95"]],
     "right": [rs["n"], rs["mean"], rs["p50"], rs["p95"]],
     "delta": [rs["n"] - ls["n"], rs["mean"] - ls["mean"],
               rs["p50"] - ls["p50"], rs["p95"] - ls["p95"]]},
    index=["count", "mean_ms", "p50_ms", "p95_ms"],
)
st.dataframe(lat_table, width="stretch")


# ---------------------------------------------------------------------------
# Module score means side-by-side
# ---------------------------------------------------------------------------
st.subheader("Module score means")
score_cols = ("gateway_prompt_score", "gateway_rag_score",
              "gateway_agency_score", "gateway_fused_score")
def _mean_scores(d: pd.DataFrame) -> dict:
    out = {}
    for c in score_cols:
        if c in d.columns:
            out[c] = float(pd.to_numeric(d[c], errors="coerce").mean() or 0.0)
        else:
            out[c] = 0.0
    return out


lm = _mean_scores(left_df)
rm = _mean_scores(right_df)
score_df = pd.DataFrame(
    {"left": [lm[c] for c in score_cols],
     "right": [rm[c] for c in score_cols],
     "delta": [rm[c] - lm[c] for c in score_cols]},
    index=[c.replace("gateway_", "").replace("_score", "") for c in score_cols],
)
st.dataframe(score_df.round(4), width="stretch")
st.bar_chart(score_df[["left", "right"]])


# ---------------------------------------------------------------------------
# Per-case diff — cases present in BOTH runs, with decision or score change
# ---------------------------------------------------------------------------
st.subheader("Per-case diff (cases present in both runs)")
if (
    "case_id" in left_df.columns and "case_id" in right_df.columns
    and not left_df.empty and not right_df.empty
):
    keep_cols = [c for c in ("case_id", "gateway_decision", "gateway_decision_band",
                              "gateway_fused_score", "gateway_miss")
                 if c in left_df.columns and c in right_df.columns]
    if "case_id" not in keep_cols:
        st.info("Need `case_id` on both sides for a meaningful diff.")
    else:
        ld = left_df[keep_cols].add_suffix("_L").rename(columns={"case_id_L": "case_id"})
        rd = right_df[keep_cols].add_suffix("_R").rename(columns={"case_id_R": "case_id"})
        merged = ld.merge(rd, on="case_id", how="inner")
        if merged.empty:
            st.info("No overlapping case_ids — different suites or no shared probes.")
        else:
            # Highlight rows where decision changed.
            if "gateway_decision_L" in merged and "gateway_decision_R" in merged:
                merged["decision_changed"] = (
                    merged["gateway_decision_L"].astype(str)
                    != merged["gateway_decision_R"].astype(str)
                )
                if "gateway_fused_score_L" in merged and "gateway_fused_score_R" in merged:
                    merged["fused_delta"] = (
                        pd.to_numeric(merged["gateway_fused_score_R"], errors="coerce")
                        - pd.to_numeric(merged["gateway_fused_score_L"], errors="coerce")
                    ).round(4)
                changed = merged[merged["decision_changed"]]
                st.metric(
                    "Cases with decision change",
                    f"{len(changed)} / {len(merged)} overlapping",
                )
                show_only_changed = st.checkbox(
                    "Show only changed cases", value=True,
                    help="Uncheck to see all overlapping cases.",
                )
                view = changed if show_only_changed else merged
                st.dataframe(view, width="stretch", hide_index=True)
else:
    st.info("Per-case diff needs `case_id` on both sides.")
