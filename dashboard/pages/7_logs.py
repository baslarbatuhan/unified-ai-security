"""Logs — explainability viewer.

Shows which signal fired (output guard / RAG chunk) and which monitoring
rule tripped, sourced from the gateway's CSVs and live alert evaluation.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from dashboard.lib.gateway_client import GatewayError, get_default_client


_RUNS = _PROJECT_ROOT / "runs"


st.set_page_config(page_title="Logs", page_icon=":scroll:", layout="wide")
st.title("Explainability logs")
st.caption("Per-decision evidence: which flag/chunk/rule caused the gateway's verdict.")


client = get_default_client()
tab_output, tab_rag, tab_alerts = st.tabs(["Output guard", "RAG guard", "Alerts"])

with tab_output:
    # Production path: output_guard/metrics_writer.py  → output_explainability_log.csv
    #   columns: run_id, case_id, target_id, flag_name, rule_or_subtype, sample
    # Batch-eval path: evaluation/run_output_guard_batch.py → output_eval_explain.csv
    #   columns: id, category, flag, evidence
    # We prefer the production file; fall back to the eval file for demos.
    _prod = _RUNS / "output_explainability_log.csv"
    _eval = _RUNS / "output_eval_explain.csv"
    p = _prod if _prod.exists() else (_eval if _eval.exists() else None)
    if p is None:
        st.info(
            "No output explainability log yet. "
            "Run `/analyze-output` calls to populate `output_explainability_log.csv`, "
            "or run `evaluation/run_output_guard_batch.py` to populate `output_eval_explain.csv`."
        )
    else:
        df = pd.read_csv(p)
        st.caption(f"{len(df)} fired flag(s) — `{p.name}`")
        # column name differs between producers
        flag_col = next((c for c in ("flag_name", "flag") if c in df.columns), None)
        if flag_col:
            flags = sorted(df[flag_col].dropna().unique().tolist())
            if flags:
                picked = st.multiselect("Flags", flags, default=flags, key="output_flags")
                df = df[df[flag_col].isin(picked)]
        st.dataframe(df, width="stretch", hide_index=True)

with tab_rag:
    # Production path: rag_guard/metrics_writer.py → rag_explainability_log.csv
    #   columns: run_id, case_id, doc_id, chunk_idx, route, embedding_score, judge_score, …
    # Batch-eval path: evaluation/build_rag_artefacts.py → rag_eval_explain.csv
    #   columns: doc_id, is_poisoned, decision, chunk_idx, chunk_judge_score, …
    _prod = _RUNS / "rag_explainability_log.csv"
    _eval = _RUNS / "rag_eval_explain.csv"
    p = _prod if _prod.exists() else (_eval if _eval.exists() else None)
    if p is None:
        st.info(
            "No RAG explainability log yet. "
            "Run the gateway with RAG docs to populate `rag_explainability_log.csv`, "
            "or run `evaluation/build_rag_artefacts.py` to populate `rag_eval_explain.csv`."
        )
    else:
        df = pd.read_csv(p)
        st.caption(f"{len(df)} chunk row(s) — `{p.name}`")
        # filter column differs between schemas
        dec_col = next((c for c in ("decision", "route") if c in df.columns), None)
        if dec_col:
            vals = sorted(df[dec_col].dropna().unique().tolist())
            picked = st.multiselect(dec_col.capitalize(), vals, default=vals, key="rag_decisions")
            df = df[df[dec_col].isin(picked)]
        st.dataframe(df, width="stretch", hide_index=True)


with tab_alerts:
    sev = st.selectbox("Min severity", ["info", "warn", "critical"], index=0)
    try:
        payload = client.get_json("/dashboard/alerts", params={"min_severity": sev}) or {}
    except GatewayError as exc:
        st.error(f"Gateway unreachable: {exc}")
        payload = {"alerts": []}

    alerts = payload.get("alerts") or []
    st.caption(f"{len(alerts)} fired rule(s) at severity ≥ {sev}")
    if not alerts:
        st.info("No rules tripped under current threshold.")
    else:
        rows = [
            {
                "rule_id": a.get("rule_id"),
                "severity": a.get("severity"),
                "title": a.get("title"),
                "detail": a.get("detail"),
                "metrics": ", ".join((a.get("metrics") or {}).keys()),
            }
            for a in alerts
        ]
        st.dataframe(rows, width="stretch", hide_index=True)
        with st.expander("Raw alert payload", expanded=False):
            st.json(alerts)
