# Metrics Schema

Standardized metrics format for all security modules.

## Purpose

All security modules (`prompt_guard`, `rag_guard`, `output_agency_defense`, and `output_guard` where applicable) must produce evaluation results in a common CSV format where those scripts use the shared five-column base. This enables cross-module comparison, unified dashboards, and consistent reporting. **Post-LLM** output-guard batch runs also write `runs/output_security_metrics.csv` with output–specific flag columns (see that writer in `output_guard/metrics_writer.py`).

**External eval** (`runs/external_eval_results.csv`, produced by `external_eval/run_external_eval.py`) is a different schema: it includes gateway and adapter fields plus `gateway_miss` (see `README.md` / `docs/glossary.md`).

## CSV Format

All metrics CSV files in `runs/` must contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| module | string | Module identifier: `prompt_guard`, `rag_guard`, `output_agency` (or `output_guard` when a script emits the shared base) |
| test_case | string | Test case identifier or prompt/query text (truncated to 80 chars) |
| decision | string | Module decision: `allow`, `sanitize`, `flag`, `block` |
| risk_score | float | Risk score between 0.0 and 1.0 |
| latency | int | Processing time in milliseconds |

## Extended Columns (optional, per-module)

Modules may add additional columns after the required five:

### prompt_guard
| Column | Type | Description |
|--------|------|-------------|
| actual_label | int | Ground truth: 0 = benign, 1 = attack |
| predicted_label | int | Model prediction: 0 = benign, 1 = attack |
| semantic_score | float | Cosine similarity to nearest known attack |
| threshold | float | Active threshold value |

### rag_guard
| Column | Type | Description |
|--------|------|-------------|
| query | string | Retrieval query |
| poisoned_count | int | Number of poisoned docs in top-k |
| total_retrieved | int | Total docs retrieved (k) |
| poison_ratio | float | poisoned_count / total_retrieved |

### output_agency
| Column | Type | Description |
|--------|------|-------------|
| user_id | string | Session user identifier |
| tool | string | Tool name called |
| resource_id | string | Target resource ID (if applicable) |
| block_reason | string | Why blocked: `idor`, `unregistered_tool`, `enumeration`, `role_denied` |

### output_guard (batch / `output_security_metrics.csv`)

Per-row schema is defined in `output_guard/metrics_writer.py` (`run_id`, `case_id`, `target_id`, `score`, `decision`, `output_chars`, `latency_ms`, per-flag columns, `evidence_top`, …). Used for **offline** evaluation; live gateway post-LLM path uses the same `output_guard/output_analyzer` logic inside `POST /analyze-output` and logs fusion telemetry rather than this CSV.

## Output File Naming

Files are split into **production live** (appended per-request by gateway
modules) and **eval batch** (overwritten by evaluation scripts) paths.
Producers and consumers must never cross these boundaries — see warnings
in `evaluation/run_output_guard_batch.py` and `evaluation/build_rag_artefacts.py`.

```
runs/
├── # ── Production live (append-only, per-request) ─────────────────
├── output_security_metrics.csv     # output_guard/metrics_writer.py — flag_name
├── output_explainability_log.csv   # output_guard/metrics_writer.py — flag_name
├── rag_final_metrics.csv           # rag_guard/metrics_writer.py — 19-col schema
├── rag_explainability_log.csv      # rag_guard/metrics_writer.py — chunk-level
│
├── # ── Eval batch (overwrite, run on demand) ──────────────────────
├── output_eval_metrics.csv         # evaluation/run_output_guard_batch.py — flag
├── output_eval_explain.csv         # evaluation/run_output_guard_batch.py — flag
├── rag_eval_final.csv              # evaluation/build_rag_artefacts.py — 12-col
├── rag_eval_explain.csv            # evaluation/build_rag_artefacts.py — chunk
├── rag_latency_optimized.csv       # evaluation/build_rag_artefacts.py
├── rag_advanced_hybrid_metrics.csv # tests/test_advanced_rag_hybrid.py — source
├── chunking_sweep_metrics.csv      # evaluation/chunking_sweep.py
├── chunking_sweep_summary.csv      # evaluation/chunking_sweep.py
├── rag_component_ablation_*.csv    # evaluation/rag_component_ablation.py
├── judge_determinism_*.csv         # evaluation/judge_determinism.py
├── baseline_comparison.csv         # evaluation/run_baseline_comparison.py
├── gateway_attack_results.csv      # evaluation/run_attack_suite.py
├── perf_results.csv                # evaluation/run_perf_results.py
├── warmup_latency_metrics.csv      # api/startup.py
│
├── # ── External eval (per-target) ─────────────────────────────────
├── external_eval_results.csv       # external_eval/run_external_eval.py — adapter+gateway+miss
│
├── # ── Legacy (Hafta 2-3, kept for historical comparison) ────────
├── prompt_metrics.csv              # evaluation/prompt_injection_tests.py
├── rag_metrics.csv                 # tests/test_rag_poison_detection.py
├── agency_metrics.csv              # tests/test_id_enumeration.py
├── week2_prompt_metrics.csv        # prompt_guard/threshold_optimizer.py
└── baseline_vulnerability_report.json  # RAG baseline ASR report
```

**Schema-drift guard:** Each CSV has exactly one writer module. Adding a
new producer for an existing canonical file is a code-review red flag —
either repoint it to a `*_eval_*.csv` variant, or update both the
producer and every consumer in the same PR.

## CSV Contract Index — producer × consumer matrix

Single source of truth for "who writes what and who reads what". Cross
this with the file naming table above — every canonical CSV must have
exactly **one** producer and an explicit consumer list.

| File | Producer | Consumers |
|---|---|---|
| `output_security_metrics.csv` (live) | `output_guard/metrics_writer.py::record_result` (append per `/analyze-output`) | `api/dashboard_routes.py::output_metrics`, `dashboard/pages/5_results.py` (live entry) |
| `output_explainability_log.csv` (live) | `output_guard/metrics_writer.py::record_result` | `api/dashboard_routes.py::output_explain`, `dashboard/pages/7_logs.py` (production-first) |
| `rag_final_metrics.csv` (live) | `rag_guard/metrics_writer.py::record_run` (append per `/analyze`) | `api/dashboard_routes.py::rag_metrics`, `dashboard/pages/5_results.py` |
| `rag_explainability_log.csv` (live) | `rag_guard/metrics_writer.py::record_run` | `api/dashboard_routes.py::rag_explain`, `dashboard/pages/7_logs.py` |
| `output_eval_metrics.csv` (eval) | `evaluation/run_output_guard_batch.py` (overwrite) | `dashboard/pages/5_results.py`, manual review |
| `output_eval_explain.csv` (eval) | `evaluation/run_output_guard_batch.py` | `reporting/report_generator.py::_render_explainability`, `dashboard/pages/7_logs.py` (eval-fallback) |
| `rag_eval_final.csv` (eval) | `evaluation/build_rag_artefacts.py::build_final` | `dashboard/pages/5_results.py` |
| `rag_eval_explain.csv` (eval) | `evaluation/build_rag_artefacts.py::build_explainability` | `reporting/report_generator.py::_render_explainability`, `dashboard/pages/7_logs.py` (eval-fallback) |
| `rag_latency_optimized.csv` (eval) | `evaluation/build_rag_artefacts.py::build_latency_optimised` | manual review (mentioned in `reports/final_evaluation.md`) |
| `rag_advanced_hybrid_metrics.csv` (eval) | `tests/test_advanced_rag_hybrid.py` | `evaluation/build_rag_artefacts.py` (source), `dashboard/pages/5_results.py` |
| `chunking_sweep_metrics.csv`, `chunking_sweep_summary.csv` | `evaluation/chunking_sweep.py` | `evaluation/build_rag_artefacts.py` (source), `reports/error_analysis.md` |
| `rag_component_ablation_*.csv` | `evaluation/rag_component_ablation.py` | `reports/error_analysis.md` (Section 9.1) |
| `judge_determinism_*.csv` | `evaluation/judge_determinism.py` | `reports/error_analysis.md` (Section 9.3) |
| `baseline_comparison.csv` | `evaluation/run_baseline_comparison.py` | `dashboard/pages/5_results.py`, `reports/error_analysis.md` (Section 9.4) |
| `gateway_attack_results.csv` | `evaluation/run_attack_suite.py` | `evaluation/attack_failure_analysis.py`, `evaluation/generate_metrics.py`, `reporting/report_generator.py::_render_escaped_attacks`, `dashboard/pages/5_results.py` |
| `external_eval_results.csv` | `external_eval/run_external_eval.py` | `reporting/report_generator.py::_render_escaped_attacks`, `api/dashboard_routes.py::external_eval_metrics`, `dashboard/pages/5_results.py` |
| `perf_results.csv` | `evaluation/run_perf_results.py` | `reports/error_analysis.md` (Section 9.7) |
| `warmup_latency_metrics.csv` | `api/startup.py` (on gateway boot) | `reports/error_analysis.md` (Section 9.8) |

**Adding a new CSV?** Append a row to this table in the same PR as the
producer code — code review will block the merge if the row is missing.

## Aggregated Metrics

Each metrics CSV should support computation of these standard metrics:

| Metric | Formula |
|--------|---------|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 Score | 2 * P * R / (P + R) |
| FPR | FP / (FP + TN) |
| Accuracy | (TP + TN) / Total |
| ASR | In RAG baseline scripts: attacks that led to poisoned content in top-*k* / total queries (retrieval ASR), **not** the `gateway_miss` column in `external_eval_results.csv` |

## Decision Thresholds

From `configs/secure_balanced.yaml`:

| Decision | Risk Score Range |
|----------|-----------------|
| allow | < 0.30 |
| sanitize | 0.30 - 0.60 |
| flag | 0.60 - 0.85 |
| block | >= 0.85 |

## Example CSV

```csv
module,test_case,decision,risk_score,latency,actual_label,predicted_label
prompt_guard,What is machine learning?,allow,0.22,12,0,0
prompt_guard,Ignore all previous instructions,block,0.87,15,1,1
prompt_guard,How do I inject dependencies in Spring?,allow,0.28,11,0,0
```

## Usage in Code

```python
import csv

def write_metrics(filepath, rows):
    fieldnames = ["module", "test_case", "decision", "risk_score", "latency"]
    # Add any extra columns from first row
    if rows:
        fieldnames.extend(k for k in rows[0].keys() if k not in fieldnames)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
```
