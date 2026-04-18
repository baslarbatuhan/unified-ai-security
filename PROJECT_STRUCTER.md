# Unified AI Security Gateway - Project Structure

This file documents the current repository layout and where each major feature lives.

## High-Level Flow

```
Client -> api/api_main.py (/analyze)
      -> fusion_gateway/engine.py
      -> [prompt_guard, rag_guard, output_agency_defense]
      -> fused decision (allow/sanitize/flag/block)
```

## Repository Tree (Current)

```
unified-ai-security/
├── api/
│   ├── __init__.py
│   ├── api_main.py
│   ├── health.py
│   ├── security_gateway.py
│   ├── security_selfcheck.py
│   └── startup.py
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
├── infra/
│   └── docker-compose.yml
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
├── reports/
├── runs/
├── schemas/
│   └── risk_schema.py
├── tests/
│   ├── conftest.py
│   ├── test_advanced_rag_poisoning.py
│   ├── test_agency_attack_scenarios.py
│   ├── test_behavior_monitor.py
│   ├── test_deobfuscation_pipeline.py
│   ├── test_id_enumeration.py
│   ├── test_prompt_evasion.py
│   └── test_rag_poison_detection.py
├── Dockerfile
├── PROJECT_STRUCTER.md
├── README.md
├── SETUP.md
└── requirements.txt
```

## Module Notes

- `prompt_guard`: deobfuscation + normalization + semantic/pattern detection + sanitization.
- `rag_guard`: embedding detector + LLM judge + robust retrieval risk scoring + context filtering.
- `output_agency_defense`: authz, anti-enum, param validation, behavior monitoring, secure tool execution.
- `fusion_gateway`: config-driven weight and threshold policy, max-rule override, parallel module execution.

## Generated Artifacts

- `runs/`: CSV/JSON outputs from tests and evaluation scripts.
- `reports/`: markdown/json analyses (e.g., attack failure analysis).

Common generated files include:
- `runs/gateway_attack_results.csv`
- `runs/gateway_attack_summary.json`
- `runs/latency_breakdown.csv`
- `runs/rag_weight_analysis.csv`
- `runs/prompt_threshold_tuning.csv`
- `runs/agency_behavior_weight_analysis.csv`
- `reports/attack_failure_analysis.md`
