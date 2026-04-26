# Unified AI Security Gateway - Project Structure

This file documents the current repository layout and where each major feature lives.

## High-Level Flow

**Input-only path (pre-LLM):**

```
Client -> api/api_main.py  POST /analyze
      -> security_gateway.SecurityGateway.analyze()
      -> fusion_gateway.engine.FusionEngine.analyze()
      -> [prompt_guard, rag_guard, output_agency]
      -> fused decision; output_score = 0.0
```

**Post-LLM path (includes model completion):**

```
Client -> api/api_main.py  POST /analyze-output
      -> SecurityGateway.analyze_with_output()
      -> FusionEngine.analyze_with_output()
      -> [prompt_guard, rag_guard, output_agency, output_guard]
      -> fused decision; output_score from output_guard module risk
```

**Observability:** `schemas/telemetry_schema` events + `logs/system_telemetry.jsonl` (typical). Dashboard under `api/dashboard_routes.py` serves JSON for the static UI in `dashboard/`.

## Repository Tree (Current)

```
unified-ai-security/
├── api/
│   ├── __init__.py
│   ├── api_main.py          # /analyze, /analyze-output, /health
│   ├── dashboard_routes.py  # read-only /dashboard/* (consumed by Streamlit UI)
│   ├── routes_runs.py       # /runs (GET history + summary; POST /runs/start spawns eval subprocess)
│   ├── routes_reports.py    # /reports (GET list/get/download; POST /reports/regenerate)
│   ├── routes_targets.py    # /targets (GET list/detail; POST/DELETE CRUD on targets.yaml)
│   ├── middleware.py        # rate limiting
│   ├── health.py
│   ├── security_gateway.py
│   ├── security_selfcheck.py
│   └── startup.py
├── dashboard/               # Streamlit UI (app.py + pages/)
├── configs/
│   ├── __init__.py
│   ├── policy_thresholds.py
│   └── secure_balanced.yaml
├── datasets/
│   ├── injection_prompts/
│   ├── output_agency_attacks/
│   └── poisoned_corpus/
├── evaluation/
│   ├── ablation_analysis.py
│   ├── agency_llm_stress_test.py
│   ├── attack_failure_analysis.py
│   ├── behavior_weight_calibration.py
│   ├── fusion_threshold_optimization.py
│   ├── generate_attack_matrix.py
│   ├── generate_metrics.py
│   ├── measure_latency_breakdown.py
│   ├── prompt_injection_tests.py
│   ├── rag_weight_optimization.py
│   ├── run_attack_suite.py
│   ├── run_experiments.py
│   ├── security_healthcheck.py
│   ├── tune_agency_behavior_weights.py
│   └── tune_prompt_threshold.py
├── fusion_gateway/
│   ├── __init__.py
│   └── engine.py
├── external_eval/          # run_external_eval.py, adapters, targets.yaml
├── infra/
│   └── docker-compose.yml
├── output_guard/
│   ├── __init__.py
│   ├── output_analyzer.py
│   └── metrics_writer.py
├── output_agency_defense/
│   ├── anti_enum_guard.py
│   ├── behavior_monitor.py
│   ├── behavior_risk_model.py
│   ├── coverage_check.py
│   ├── error_policy.py
│   ├── guard_registry.py
│   ├── object_authz_guard.py
│   ├── parameter_validation.py
│   ├── resource_registry.py
│   ├── risk_scoring.py
│   ├── sandbox_executor.py
│   ├── secure_tool_wrapper.py
│   ├── sequential_probe_detector.py
│   └── tool_call_simulator.py
├── prompt_guard/
│   ├── deobfuscator.py
│   ├── pattern_detector.py
│   ├── pipeline.py
│   ├── prompt_normalizer.py
│   ├── prompt_sanitizer.py
│   ├── risk_scoring.py
│   ├── semantic_evaluator_v1.py
│   └── threshold_optimizer.py
├── rag_guard/
│   ├── build_safe_context.py
│   ├── context_analysis.py
│   ├── context_filter.py
│   ├── llm_judge.py
│   ├── pipeline.py
│   ├── poison_detector.py
│   ├── rag_baseline.py
│   ├── retrieval_risk_score.py
│   └── risk_scoring.py
├── reports/                 # e.g. attack analyses, ops_notes.md
├── runs/
├── schemas/                 # risk + telemetry
│   ├── risk_schema.py
│   └── telemetry_schema.py
├── tests/                   # phase* and feature tests (see tree with `ls tests/`)
├── Dockerfile
├── PROJECT_STRUCTURE.md
├── README.md
├── SETUP.md
└── requirements.txt
```

## Module Notes

- `prompt_guard`: deobfuscation + normalization + semantic/pattern detection + sanitization.
- `rag_guard`: embedding detector + LLM judge + robust retrieval risk scoring + context filtering.
- `output_agency_defense`: authz, anti-enum, param validation, behavior monitoring, secure tool execution; fusion module name `output_agency` (tool-call and prompt scan when a tool is present).
- `output_guard`: post-hoc scoring of **model text**; used in `FusionEngine.analyze_with_output`, HTTP `POST /analyze-output`, and batch `evaluation/` / metrics writers. Not a replacement for tool-use safety — complements `output_agency_defense`.
- `fusion_gateway`: config-driven weight and threshold policy, max-rule override, parallel input-side module execution; fourth module only on the `analyze_with_output` code path.
- `external_eval`: drive external chatbot targets, record `gateway_decision` and `gateway_miss` in `runs/external_eval_results.csv`.

## Generated Artifacts

- `runs/`: CSV/JSON outputs from tests and evaluation scripts.
- `reports/`: markdown/json analyses (e.g., attack failure analysis, `ops_notes.md`).

Common generated files include:
- `runs/gateway_attack_results.csv`
- `runs/gateway_attack_summary.json`
- `runs/latency_breakdown.csv`
- `runs/rag_weight_analysis.csv`
- `runs/prompt_threshold_tuning.csv`
- `runs/agency_behavior_weight_analysis.csv`
- `runs/output_security_metrics.csv`   # batch output-guard runs
- `runs/external_eval_results.csv`        # includes `gateway_miss` column
- `logs/system_telemetry.jsonl`         # if telemetry sink enabled
- `reports/attack_failure_analysis.md`
- `reports/ops_notes.md`
