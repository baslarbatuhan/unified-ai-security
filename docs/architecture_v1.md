# Architecture v1

This document matches the current gateway as implemented in the repository (see also `README.md` and `PROJECT_STRUCTURE.md`).

## HTTP surface

| Endpoint | Purpose |
|----------|---------|
| `POST /analyze` | **Pre-LLM** screening. Runs prompt guard, RAG guard, and output-agency (tool path) in parallel, then fuses. Response field `output_score` is always `0.0` (no model completion analyzed). |
| `POST /analyze-output` | **Post-LLM** screening. Same request body as `/analyze` **plus** required `model_output`. Re-runs the three input-side modules and runs **output guard** on the completion, then fuses all four. `output_score` reflects the output-guard module risk. |
| `GET /health` | Liveness / fusion readiness checks. |
| `GET /dashboard/*` | Read-only JSON consumed by the Streamlit UI (summary, recent fusion decisions, CSV tails, breakers, alert rules). |
| `GET /runs/*`, `GET /reports/*`, `GET /targets/*` | Run history, generated reports, and external_eval targets — also consumed by the Streamlit UI. |
| `POST /runs/start` | Spawns `external_eval/run_external_eval.py` as a background subprocess; returns `{run_id, command}`. Triggered by the Run test page. |
| `POST /reports/regenerate` | Re-runs `reporting/report_generator.py`, overwriting `reports/chatbot_security_report.md`. Triggered by the Reports page. |
| `POST /targets`, `DELETE /targets/{id}` | Mutating CRUD on `external_eval/targets.yaml`; the Streamlit Targets page is the only caller. |

Rate limiting is applied in `api/middleware.py` (applies to dashboard routes as well except where exempted in code).

## Fusion engine paths

- **`FusionEngine.analyze()`** — three modules: `prompt_guard`, `rag_guard`, `output_agency`. Used by `SecurityGateway.analyze()` and `POST /analyze`.
- **`FusionEngine.analyze_with_output()`** — four modules: same three + `output_guard` evaluated on `model_output`. Used by `SecurityGateway.analyze_with_output()` and `POST /analyze-output`.

`output_agency` is about **structured tool calls** and role/authz; `output_guard` is about **free-text model output** (PII-like patterns, key-like strings, unsafe instructions, injection phrasing, off-allowlist URLs). They are complementary.

## Telemetry

`SecurityGateway` emits `RequestEvent` and `FusionDecisionEvent` (the latter includes `output_score` when the post-LLM path runs). The dashboard’s “live” panels read recent events (typically from `logs/system_telemetry.jsonl` via `api/dashboard_routes.py`). Full-file read behavior and future scaling options are described in `reports/ops_notes.md`.

## External evaluation

`external_eval/run_external_eval.py` produces `runs/external_eval_results.csv` with a `gateway_miss` column: it flags when the **expected** decision is `block` or `sanitize` but the gateway returned `allow`. This is a **protector** metric, not the same as RAG retrieval “attack success rate” (ASR) in `rag_guard/rag_baseline.py`.

## Explainability — two parallel pipelines

Per-decision evidence is written by two independent producers that **must
not be confused**:

| Aspect | Live (per-request) | Batch eval |
|---|---|---|
| Files | `runs/output_explainability_log.csv`, `runs/rag_explainability_log.csv` | `runs/output_eval_explain.csv`, `runs/rag_eval_explain.csv` |
| Producer | `output_guard/metrics_writer.py`, `rag_guard/metrics_writer.py` (append per gateway request) | `evaluation/run_output_guard_batch.py`, `evaluation/build_rag_artefacts.py` (overwrite on demand) |
| Schema | Wider — includes `run_id`, `target_id`, `route_*` (live) | Narrower — eval-focused (`flag` instead of `flag_name`, etc.) |
| Consumed by | **Streamlit Logs page** (`dashboard/pages/7_logs.py`) | **Markdown report** (`reporting/report_generator.py::_render_explainability`) |
| Triggered by | Real `/analyze` / `/analyze-output` traffic | `python evaluation/run_output_guard_batch.py` etc. |

Both pipelines are intentional: the live log is a forensic stream tied
to real telemetry events; the eval CSV is a deterministic, dataset-bound
snapshot suitable for embedding in a thesis report. The Logs page
auto-detects which file is present (production-first, eval-fallback) so
demos work end-to-end even before the gateway has served real traffic.

If the regenerated report shows
`No runs/output_eval_explain.csv — run python evaluation/run_output_guard_batch.py …`,
that is the eval pipeline talking — running real `/analyze-output`
requests will **not** populate that file (it populates the live log
instead, which the dashboard renders).

## Escape vs downgrade — alignment

The report's "Escaped attacks" section uses the same predicate as
`external_eval/run_external_eval.py::_classify_gateway_miss`:

```
escape  = expected ∈ {block, sanitize} AND got = allow
```

Cases where the gateway downgraded a `block`-expected case to
`sanitize` or `flag` are rendered in a separate sub-table — they are
**not** counted as misses (protection still fired) but are surfaced so
threshold drift between design intent and gateway behaviour stays
auditable.
