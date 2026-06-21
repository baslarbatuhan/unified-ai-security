# Unified AI Security Gateway — Project Structure

Where each major feature lives and how data flows through the repo.

## High-Level Flow

**Input-only path (pre-LLM):**

```
Client -> api/api_main.py  POST /analyze
      -> security_gateway.SecurityGateway.analyze()
      -> fusion_gateway.engine.FusionEngine.analyze()
      -> [prompt_guard, rag_guard, output_agency]  (parallel)
      -> fused decision; output_score = 0.0
```

**Post-LLM path:**

```
Client -> POST /analyze-output
      -> SecurityGateway.analyze_with_output()
      -> FusionEngine.analyze_with_output()
      -> [prompt_guard, rag_guard, output_agency, output_guard]
      -> fused decision; output_score from output_guard
```

**External eval (live targets, Sprint 11 — gateway first):**

```
attack_suites.load_suite()
      -> SecurityGateway.analyze()          # verdict first
      -> fusion_gateway.firewall_policy     # passthrough or firewall
      -> adapter.send() if forwarded        # api | web | tools_local | mock
      -> runs/<run_id>/results.csv
```

**Observability:** `schemas/telemetry_schema.py` events → `logs/system_telemetry.jsonl`; per-decision `runs/<run_id>/decision_trace.csv`; dashboard reads via `api/dashboard_routes.py` and `utils/run_manifest.py`.

## Repository Tree

```
unified-ai-security/
├── api/
│   ├── api_main.py              # FastAPI app entry
│   ├── security_gateway.py      # analyze() / analyze_with_output()
│   ├── dashboard_routes.py      # GET /dashboard/* (Streamlit polls)
│   ├── routes_runs.py           # GET /runs/*, POST /runs/start
│   ├── routes_reports.py        # GET/POST /reports/*
│   ├── routes_targets.py        # GET/POST/DELETE /targets
│   ├── routes_decisions.py      # GET /decisions/{run_id}/{case_id}/trace
│   ├── routes_secrets.py        # vault CRUD (dashboard Targets)
│   ├── middleware.py            # rate limiting
│   ├── rate_limiter.py
│   ├── health.py
│   ├── startup.py               # warm-up, self-check hooks
│   └── security_selfcheck.py
├── dashboard/
│   ├── app.py
│   ├── lib/
│   │   ├── gateway_client.py    # HTTP client for gateway API
│   │   ├── components.py        # shared UI widgets
│   │   ├── target_presets.py    # OpenAI, Anthropic, Open WebUI, … presets
│   │   └── recommendations.py   # Home page score + rule-based recos
│   └── pages/
│       ├── 0_admin.py
│       ├── 1_home.py            # executive summary, FP/FN recos
│       ├── 2_targets.py         # CRUD + presets + secrets vault
│       ├── 3_run_test.py        # single + suite; firewall toggle; coupled weights
│       ├── 4_live_monitor.py
│       ├── 5_results.py         # per-run analytics + firewall strip
│       ├── 6_reports.py
│       ├── 7_logs.py            # explainability CSVs
│       └── 8_compare_runs.py    # passthrough vs firewall diff
├── configs/
│   ├── secure_balanced.yaml     # fusion weights, RAG judge, thresholds
│   ├── timeout_config.yaml      # fail-closed module budgets
│   ├── service_limits.yaml      # rate limits, eval parallelism
│   ├── security_score_weights.yaml
│   ├── alert_rules.yaml
│   └── policy_thresholds.py
├── datasets/
│   ├── injection_prompts/
│   ├── poisoned_corpus/         # basic + advanced poison JSON
│   ├── output_agency_attacks/
│   ├── prompt_regression_set.json
│   └── output_guard_eval_set.json
├── docs/
│   ├── architecture_v1.md       # HTTP routes, adapter axis
│   ├── glossary.md              # gateway_miss vs ASR
│   ├── threat_model.md
│   ├── prompt_injection_threat_model.md
│   ├── rag_threat_model.md
│   ├── agency_threat_model.md
│   ├── security_objectives.md
│   ├── design_decisions.md
│   └── manual_test_sprints_0_7.md
├── evaluation/                  # batch scripts (24 modules)
├── external_eval/
│   ├── run_external_eval.py     # CLI + CSV writer; --firewall; gateway-first
│   ├── attack_suites.py         # prompt_injection, rag_poisoning, agency_social
│   ├── targets.yaml             # target registry (compose bind-mount)
│   ├── api_adapter.py           # REST targets + Open WebUI path
│   ├── web_adapter.py           # Playwright
│   ├── tool_adapter.py          # tools_local
│   ├── openwebui_chat.py        # Open WebUI chat UUID + turn register
│   ├── secrets_store.py         # runs/.secrets.yaml vault
│   └── adapter_factory.py
├── fusion_gateway/
│   ├── engine.py                # FusionEngine, parallel module dispatch
│   ├── firewall_policy.py       # decide(), resolve_policy(), blocked response
│   └── fallback_handler.py
├── prompt_guard/                # deobfuscator, semantic, pattern, sanitizer
├── rag_guard/                   # poison_detector, llm_judge, chunk_router, filter
├── output_agency_defense/       # authz, anti-enum, param validation, behavior
├── output_guard/                # output_analyzer, metrics_writer
├── tools/                       # weather, stock, calculator (tools_local)
├── utils/
│   ├── run_manifest.py          # runs/_registry.jsonl, manifest.json
│   ├── config_builder.py        # runs/<run_id>/config_used.yaml snapshots
│   ├── log_sanitizer.py
│   └── rate_limiter.py
├── schemas/
│   ├── risk_schema.py           # AnalyzeRequest, ConfigOverrides, …
│   ├── target_schema.py         # TargetConfig, TargetPolicy
│   └── telemetry_schema.py
├── scripts/
│   ├── export_webui_state.py    # Playwright storage_state for web targets
│   ├── generate_thesis_report.py
│   ├── generate_thesis_figures.py
│   ├── backfill_run_registry.py
│   ├── seed_dashboard.py
│   └── smoke_sprints_0_7.sh
├── infra/
│   ├── docker-compose.yml       # gateway, dashboard, chroma, ollama, open-webui
│   ├── docker-compose.override.yml
│   ├── docker-compose.gpu.yml
│   ├── compose-config-safe.sh
│   └── .env.example
├── tests/                       # 609 pytest cases
├── Dockerfile
├── requirements.txt
├── README.md
└── SETUP.md
```

## Module Notes

| Package | Role |
|---------|------|
| `prompt_guard` | Deobfuscation, NFKC normalize, BGE-M3 semantic + regex, sanitize band |
| `rag_guard` | Embedding pre-filter, chunked LLM judge (Ollama), hybrid score, context filter |
| `output_agency_defense` | Tool-call authz, sequential probe detection (singleton state), param schema |
| `output_guard` | Post-LLM text: PII, secrets, unsafe instructions, off-allowlist URLs |
| `fusion_gateway` | Weighted sum + max-rule override; timeout fail-closed |
| `external_eval` | Adapter axis; gateway-first loop; `gateway_miss`; firewall CSV columns |

## Target Adapters (`external_eval/`)

| `type` | Adapter | Use case |
|--------|---------|----------|
| `api` | `APIAdapter` | REST POST/GET (Open WebUI completions, cloud APIs) |
| `web` | `WebAdapter` | Playwright-driven chat UIs |
| `tools_local` | `ToolAdapter` | Gateway pre-screen + real tool invoke |
| `mock` | `MockAdapter` | CI / offline |

Built-in targets in `targets.yaml` (enable/disable as needed):

| id | type | Notes |
|----|------|-------|
| `mock_echo` | mock | CI smoke |
| `local_tools` | tools_local | agency_social + tool execution |
| `open_webui_api` | api | REST `/api/chat/completions` |
| `open_webui_web` | web | Playwright + `storage_state_path` |
| `internal_chatbot_api` | api | template for private REST chatbots |

Open WebUI: `open_webui_api` (REST + `openwebui_chat.py`) and `open_webui_web` (Playwright). Session export: `scripts/export_webui_state.py`.

## Target Policy (`schemas/target_schema.py`)

Each target may define:

```yaml
policy:
  mode: passthrough | firewall    # default passthrough
  on_block: drop                  # MVP: block always drops in firewall mode
  on_sanitize: forward_marked     # prepend warning banner when forwarded
```

CLI `--firewall` promotes effective mode to firewall for that run. Dashboard **Run test** suite mode exposes the same toggle.

## Generated Artifacts (`runs/`, `logs/`, `reports/` — gitignored)

Per eval run (`runs/<run_id>/`):

- `results.csv` — case-level gateway + adapter + firewall columns
- `decision_trace.csv` — fusion audit row per case
- `config_used.yaml` — reproducible config snapshot
- `manifest.json`, `runner.log`, `status.json`

Registry:

- `runs/_registry.jsonl` — index for dashboard pickers
- `runs/external_eval_results.csv` — cross-run aggregate

Secrets (never commit):

- `runs/.secrets.yaml` — dashboard Targets vault (mode 0600)
- `runs/open_webui_state.json` — Playwright session for web targets

Telemetry:

- `logs/system_telemetry.jsonl`

Common eval outputs:

- `runs/gateway_attack_results.csv`
- `runs/rag_final_metrics.csv`, `runs/rag_explainability_log.csv`
- `runs/output_security_metrics.csv`, `runs/output_explainability_log.csv`
- `reports/attack_failure_analysis.md`

## Key CSV Columns (external eval)

| Column | Meaning |
|--------|---------|
| `gateway_decision` | allow / sanitize / block |
| `gateway_miss` | 1 if expected block/sanitize but got allow |
| `mode` | passthrough or firewall (effective per run/target) |
| `firewall_action` | forward / forward_marked / drop |
| `forwarded_to_target` | 1 if adapter.send was invoked |
| `adapter_state` | ok / blocked_by_gateway_pre / target_error |
| `adapter_error` | `blocked_by_gateway` when firewall dropped the prompt |
| `tool_executed` | 1 if tools_local invoked after allow/sanitize |

## Dashboard Run Test (Sprint 10)

Reactive UI zones (outside submit form):

1. Target + suite pickers
2. Module checkboxes → disabled modules excluded from weight sliders and normalization
3. Coupled fusion sliders (sum = 1.00)
4. Suite: firewall toggle + max attacks; single: prompt + model output

Config snapshot written to `runs/<run_id>/config_used.yaml` on every submit.

## Related Docs

- `README.md` — quick start
- `SETUP.md` — install, Docker, Open WebUI, troubleshooting
- `docs/architecture_v1.md` — HTTP routes, adapter axis, explainability pipelines
- `docs/glossary.md` — gateway_miss vs ASR
