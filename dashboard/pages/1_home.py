"""Home — Executive overview (Hafta 13).

Top-of-funnel page for non-technical users. Reads the run registry +
the gateway's read-only telemetry feed, computes a composite security
score, and renders four blocks:

    1. Header: security_score / risk level / coverage caption
    2. KPI strip: miss rate / block rate / latency / P / R / F1
    3. Attack distribution + per-module success rate
    4. Critical findings + auto-generated recommendations

Technical health (circuit breakers, raw telemetry, /health checks) lives
on `0_admin.py` so this page never feels like an ops console.

All number-crunching is delegated to:
  * `utils.run_manifest.read_registry` — newest-first run metadata
  * `dashboard.lib.recommendations` — score formula + reco rules
  * `dashboard.lib.components` — UI primitives

Data sources, in order:
  * `/dashboard/summary`            — block / sanitize / allow rates + latency
  * `runs/_registry.jsonl`          — recent runs, per-run results.csv
  * `runs/rag_final_metrics.csv`    — routing savings + module latency
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from dashboard.lib.components import (
    DECISION_COLORS,
    executive_summary_panel,
    recommendation_panel,
)
from dashboard.lib.gateway_client import GatewayError, get_default_client
from dashboard.lib.recommendations import (
    composite_security_score,
    generate_recommendations,
)
from utils.run_manifest import RESULTS_FILENAME, read_registry


_RUNS_DIR = _PROJECT_ROOT / "runs"
_LEGACY_AGG_CSV = _RUNS_DIR / "external_eval_results.csv"
_RAG_METRICS_CSV = _RUNS_DIR / "rag_final_metrics.csv"
_RECENT_RUN_WINDOW = 20  # max runs aggregated into the headline score


st.set_page_config(page_title="Executive overview", page_icon=":bar_chart:", layout="wide")
st.title("Executive overview")
st.caption(
    "Single-glance view of how the gateway is performing — built for "
    "non-technical reviewers. For circuit breakers, raw telemetry and "
    "health checks, open the **Admin** page."
)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
client = get_default_client()
try:
    summary = client.get_json("/dashboard/summary") or {}
except GatewayError as exc:
    st.error(f"Gateway unreachable: {exc}")
    st.stop()


def _load_recent_runs(limit: int) -> List[Dict[str, Any]]:
    return read_registry(_RUNS_DIR, limit=limit)


def _per_run_rows(entry: Dict[str, Any]) -> pd.DataFrame:
    """Prefer the per-run results.csv; fall back to filtering the legacy
    aggregate. Empty DataFrame if neither has data for this run."""
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


def _aggregate_metrics(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine the last N runs into the metrics dict that the score
    function + recommendations engine consume.

    Block-class confusion matrix (precision / recall / F1) uses the
    aggregated expected_decision vs gateway_decision counts so a single
    poorly-tuned run doesn't dominate. Band counts come from the new
    `gateway_decision_band` column when present.
    """
    if not runs:
        return {"n_recent_runs": 0}

    frames: List[pd.DataFrame] = []
    for r in runs:
        df = _per_run_rows(r)
        if not df.empty:
            frames.append(df)
    if not frames:
        return {"n_recent_runs": len(runs)}

    df = pd.concat(frames, ignore_index=True)
    total = len(df)

    # Decision distribution.
    dec = df.get("gateway_decision", pd.Series(dtype=str)).astype(str).str.lower()
    block_count = int((dec == "block").sum())
    sanitize_count = int((dec == "sanitize").sum())
    allow_count = int((dec == "allow").sum())
    miss = pd.to_numeric(df.get("gateway_miss", pd.Series(dtype=int)),
                         errors="coerce").fillna(0)
    miss_count = int(miss.sum())

    # Confusion matrix on block class.
    exp = df.get("expected_decision", pd.Series(dtype=str)).astype(str).str.lower()
    tp = int(((exp == "block") & (dec == "block")).sum())
    fn = int(((exp == "block") & (dec != "block")).sum())
    fp = int(((exp != "block") & (dec == "block")).sum())
    tn = int(((exp != "block") & (dec != "block")).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0

    # Latency aggregation: mean of `gateway_latency_ms`. The "breach
    # rate" needs configured budgets to compute precisely; without that
    # we approximate by counting calls > 2σ above the mean.
    lat = pd.to_numeric(df.get("gateway_latency_ms", pd.Series(dtype=float)),
                        errors="coerce").dropna()
    avg_latency = float(lat.mean()) if not lat.empty else 0.0
    if len(lat) >= 5:
        sigma_threshold = lat.mean() + 2 * (lat.std(ddof=0) or 0.0)
        latency_breach_rate = float((lat > sigma_threshold).mean())
    else:
        latency_breach_rate = 0.0

    # decision_band balance (flag-tier vs confident block).
    band = df.get("gateway_decision_band", pd.Series(dtype=str)).astype(str).str.lower()
    flag_band_count = int((band == "flag").sum())
    block_band_count = int((band == "block").sum())

    # Attack distribution by `suite`.
    suite_dist = (
        df.get("suite", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
    )

    # Per-module mean scores (across all rows).
    mod_means: Dict[str, float] = {}
    for col, label in (
        ("gateway_prompt_score", "prompt_guard"),
        ("gateway_rag_score", "rag_guard"),
        ("gateway_agency_score", "output_agency"),
    ):
        if col in df.columns:
            mod_means[label] = float(
                pd.to_numeric(df[col], errors="coerce").mean() or 0.0
            )

    # Agency engagement signal — % of cases where agency actually scored.
    agency_scores = pd.to_numeric(
        df.get("gateway_agency_score", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0.0)
    avg_agency_score = float(agency_scores.mean()) if not agency_scores.empty else 0.0
    n_agency_cases = int((agency_scores > 0).sum())

    # Routing savings — joined from rag_final_metrics.csv. Per-run join
    # is cheaper than reading the whole file once if there are few runs.
    routing_savings_pct: Optional[float] = None
    if _RAG_METRICS_CSV.exists():
        try:
            rag_df = pd.read_csv(_RAG_METRICS_CSV)
            rids = {r.get("run_id") for r in runs if r.get("run_id")}
            scoped = rag_df[rag_df["run_id"].astype(str).isin(rids)] if rids else pd.DataFrame()
            if not scoped.empty and "routing_savings_pct" in scoped.columns:
                routing_savings_pct = float(
                    pd.to_numeric(scoped["routing_savings_pct"], errors="coerce").mean() or 0.0
                )
        except Exception:  # noqa: BLE001
            routing_savings_pct = None

    return {
        "n_recent_runs": len(runs),
        "n_rows": total,
        "miss_rate": (miss_count / total) if total > 0 else 0.0,
        "block_rate": (block_count / total) if total > 0 else 0.0,
        "sanitize_rate": (sanitize_count / total) if total > 0 else 0.0,
        "allow_rate": (allow_count / total) if total > 0 else 0.0,
        "avg_latency_ms": avg_latency,
        "latency_breach_rate": latency_breach_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_rate": fp_rate,
        "flag_band_count": flag_band_count,
        "block_band_count": block_band_count,
        "suite_distribution": suite_dist,
        "module_mean_scores": mod_means,
        "avg_agency_score": avg_agency_score,
        "n_agency_cases": n_agency_cases,
        "routing_savings_pct": routing_savings_pct,
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
runs = _load_recent_runs(_RECENT_RUN_WINDOW)
agg = _aggregate_metrics(runs)
score = composite_security_score(agg)
agg["security_score"] = score

if not runs:
    st.warning(
        "No runs registered yet. Launch a suite from the **Run test** page "
        "to populate this view."
    )
    # We still render the empty header so the page doesn't look broken
    # in fresh installs — security_score 0, all metrics 0.
    executive_summary_panel(agg)
    st.stop()

# ---------------------------------------------------------------------------
# 1) Headline
# ---------------------------------------------------------------------------
executive_summary_panel(agg)


# ---------------------------------------------------------------------------
# 2) Attack distribution + per-module success
# ---------------------------------------------------------------------------
ad_col, mod_col = st.columns(2)

with ad_col:
    st.subheader("Attack distribution")
    suite_dist = agg.get("suite_distribution") or {}
    if suite_dist:
        sd_df = pd.DataFrame(
            {"suite": list(suite_dist.keys()), "count": list(suite_dist.values())}
        ).set_index("suite")
        st.bar_chart(sd_df)
    else:
        st.caption("No suite metadata in recent runs.")

with mod_col:
    st.subheader("Module average risk (last runs)")
    mod_means = agg.get("module_mean_scores") or {}
    if mod_means:
        mod_df = pd.DataFrame(
            {"module": list(mod_means.keys()), "avg_risk": list(mod_means.values())}
        ).set_index("module")
        st.bar_chart(mod_df)
        st.caption(
            "Higher average risk = module is firing more often. Compare with "
            "the per-module latency on the **Live monitor** page."
        )
    else:
        st.caption("No per-module score data yet.")


# ---------------------------------------------------------------------------
# 3) Critical findings — recent false negatives
# ---------------------------------------------------------------------------
st.subheader("Critical findings (recent runs)")
fn_rows = []
for r in runs[:5]:  # top 5 newest runs only — keep it scannable
    df = _per_run_rows(r)
    if df.empty:
        continue
    if "expected_decision" not in df.columns or "gateway_decision" not in df.columns:
        continue
    exp = df["expected_decision"].astype(str).str.lower()
    got = df["gateway_decision"].astype(str).str.lower()
    fns = df[(exp == "block") & (got != "block")]
    for _, row in fns.iterrows():
        fn_rows.append({
            "run_id": r.get("run_id", "")[-12:],
            "case_id": row.get("case_id", ""),
            "expected": row.get("expected_decision", ""),
            "got": row.get("gateway_decision", ""),
            "fused_risk": row.get("gateway_fused_score", ""),
            "category": row.get("category", ""),
        })

if fn_rows:
    st.dataframe(pd.DataFrame(fn_rows).head(20), width="stretch", hide_index=True)
    st.caption(
        f"{len(fn_rows)} false-negative case(s) in the last "
        f"{min(5, len(runs))} run(s). Each represents an attack the gateway "
        "let through. Investigate from the **Results > Failures** tab."
    )
else:
    st.success(
        "No false negatives in the recent runs — every attack expected to "
        "block / sanitize was caught."
    )


# ---------------------------------------------------------------------------
# 4) Recommendations
# ---------------------------------------------------------------------------
recos = generate_recommendations(agg)
recommendation_panel(recos, title="Recommendations")
if not recos:
    st.caption(
        "_System looks healthy on the metrics this dashboard tracks._"
    )


# ---------------------------------------------------------------------------
# Footer: source visibility — analysts often want to know what we read.
# ---------------------------------------------------------------------------
with st.expander("How this is calculated"):
    st.markdown(
        f"""
- **Window**: last `{agg.get('n_recent_runs', 0)}` runs from `runs/_registry.jsonl`
  (cap = {_RECENT_RUN_WINDOW}).
- **Rows aggregated**: `{agg.get('n_rows', 0)}` from per-run `results.csv`
  (fallback to `external_eval_results.csv`).
- **Security score** = `100 × (1 − miss_rate) × (1 − latency_breach_rate) ×
  (precision × recall)`. Score ≥ 85 = LOW risk, ≥ 70 MEDIUM, ≥ 50 HIGH,
  otherwise CRITICAL.
- **latency_breach_rate** ≈ fraction of calls more than 2σ over the
  window's mean (proxy until per-module budget joins land).
- **Recommendations** use the same metrics dict — thresholds in
  `dashboard/lib/recommendations.py`.
        """
    )
