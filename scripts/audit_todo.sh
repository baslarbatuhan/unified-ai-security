#!/usr/bin/env bash
cd "$(dirname "$0")/.."
paths=(
  external_eval/api_adapter.py
  external_eval/web_adapter.py
  external_eval/targets.yaml
  external_eval/run_external_eval.py
  runs/external_eval_results.csv
  api/rate_limiter.py
  utils/rate_limiter.py
  fusion_gateway/fallback_handler.py
  utils/fallback_handler.py
  configs/timeout_config.yaml
  configs/service_limits.yaml
  schemas/telemetry_schema.py
  logs/system_telemetry.jsonl
  monitoring/alert_rules.py
  configs/alert_rules.yaml
  dashboard/app.py
  tests/test_gateway.py
  tests/test_fusion.py
  reports/contribution_analysis.md
  utils/log_sanitizer.py
  output_guard/output_analyzer.py
  runs/output_security_metrics.csv
  runs/output_explainability_log.csv
  rag_guard/chunk_router.py
  runs/rag_latency_optimized.csv
  runs/rag_final_metrics.csv
  runs/rag_explainability_log.csv
  runs/prompt_stability_check.csv
  datasets/prompt_regression_set.json
  reports/agency_external_eval_design.md
  datasets/agency_demo_cases.json
  reporting/report_generator.py
  reports/chatbot_security_report.md
  reporting/summary_generator.py
  reporting/recommendation_engine.py
  runs/baseline_comparison.csv
  runs/perf_results.csv
  reports/final_evaluation.md
  utils/config_builder.py
  api/routes_runs.py
  api/routes_targets.py
  api/routes_reports.py
  dashboard/pages/1_home.py
  dashboard/pages/2_targets.py
  dashboard/pages/3_run_test.py
  dashboard/pages/4_live_monitor.py
  dashboard/pages/5_results.py
  dashboard/pages/6_reports.py
  dashboard/pages/7_logs.py
)
for f in "${paths[@]}"; do
  if [ -e "$f" ]; then
    echo "OK $f"
  else
    echo "MISS $f"
  fi
done
