# Unified AI Security Gateway

Unified gateway for defending LLM applications against:
- prompt injection / jailbreak attempts
- RAG poisoning and context manipulation
- tool misuse (IDOR, enumeration, param abuse, role misuse)
- model output risks (leaked PII/secrets, unsafe instructions, redirect/injection) via **output guard**

The system runs **input-side** module pipelines in parallel and fuses their scores into a single decision. A separate **post-LLM** path runs the same input modules **plus** output text analysis (output guard).

## Architecture

**Pre-LLM (input screening)** — client sends the user prompt and context; the gateway does *not* call the target LLM.

```
POST /analyze
    |
    +--> Prompt Guard   (deobfuscate -> normalize -> semantic/pattern -> sanitize)
    +--> RAG Guard      (poison detector -> LLM judge -> retrieval risk -> context filter)
    +--> Agency Defense (tool/authz -> anti-enum -> param validation -> behavior signals; tool-call path)
                  |
                  v
          Fusion Gateway (weighted sum + max-rule override)
                  |
                  v
      allow / sanitize / flag / block     (response: output_score = 0.0 on this path)
```

**Post-LLM (output screening)** — after the *client* calls its own LLM, the client posts the model completion for the same request shape:

```
POST /analyze-output  (requires model_output)
    |
    +--> same three input-side modules (stateless re-check)
    +--> Output Guard   (regex/entropy: PII, API-key-like tokens, unsafe text, injection, off-allowlist URLs)
                  |
                  v
          Fusion (four modules; output_score reflects output_guard contribution)
                  |
                  v
      allow / sanitize / flag / block
```

**Dashboard** is a Streamlit app (`dashboard/app.py`) that talks to the gateway over HTTP and uses both read and write routes:

- **Read-only** (polled): `GET /dashboard/*` (summary, alerts, recent-runs, breakers, metrics), `GET /runs`, `GET /reports`, `GET /targets`, `GET /health`.
- **Mutating** (button-driven): `POST /analyze`, `POST /analyze-output`, `POST /runs/start`, `POST /reports/regenerate`, `POST /targets`, `DELETE /targets/{id}`.

Run it as a sibling process to the gateway. See `reports/ops_notes.md` for operational limits and deferred work.

## Core Components

- `api/api_main.py`: FastAPI app (`POST /analyze`, `POST /analyze-output`, `GET /health`)
- `api/security_gateway.py`: `analyze()` and `analyze_with_output()`; telemetry (`FusionDecisionEvent` includes `output_score` on the post-LLM path)
- `api/dashboard_routes.py` + `api/middleware.py`: read-only dashboard JSON, rate limits
- `fusion_gateway/engine.py`: module execution + fusion; `analyze()` vs `analyze_with_output()` (adds `output_guard`)
- `prompt_guard/pipeline.py`: full prompt defense pipeline
- `rag_guard/pipeline.py`: hybrid RAG defense pipeline
- `output_agency_defense/*`: agency / tool-call protections
- `output_guard/output_analyzer.py`: post-hoc model *text* risk (used on `/analyze-output` and in batch eval)
- `external_eval/run_external_eval.py`: live-target harness; CSV includes `gateway_miss` (see glossary / ops notes)
- `schemas/risk_schema.py`: common request/response schema

## Repository Layout

```
unified-ai-security/
├── api/                 # FastAPI, dashboard routes, security gateway, middleware
├── dashboard/            # Streamlit UI (app.py + pages/)
├── configs/
├── datasets/
├── evaluation/           # attack suite, metrics, ablations
├── external_eval/      # adapters + run_external_eval (targets, gateway_miss CSV)
├── fusion_gateway/
├── output_guard/         # output_analyzer + metrics writer
├── output_agency_defense/
├── prompt_guard/
├── rag_guard/
├── schemas/
├── tests/
├── infra/
├── runs/                 # generated CSV/JSON (metrics, attack results, telemetry log path)
└── reports/              # generated analyses; ops_notes.md = ops/deployment notes
```

## Quick Start

### Option A — Docker (recommended, no host setup)

`docker compose up` brings the whole stack online with the same surface
as a local run: gateway on `:8000`, dashboard on `:8501`, plus Chroma
and Ollama for RAG/judge. State (target YAML, generated CSVs) is
bind-mounted to the host so it survives container restarts.

```bash
git clone https://github.com/<user>/unified-ai-security.git
cd unified-ai-security/infra
cp .env.example .env          # fill in HF_TOKEN / API keys if you have them
docker compose up -d
# gateway   → http://localhost:8000
# dashboard → http://localhost:8501
docker compose logs -f gateway   # warmup takes ~60-90 s on first boot
```

GPU (NVIDIA host with the Container Toolkit):

```bash
docker compose -f docker-compose.yml \
               -f docker-compose.override.yml \
               -f docker-compose.gpu.yml up -d
```

Stop with `docker compose down` (`-v` also drops the chroma/ollama
volumes; omit if you want the LLM weights to survive).

### Option B — Local Python venv

```bash
git clone https://github.com/<user>/unified-ai-security.git
cd unified-ai-security
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env
```

Run gateway:

```bash
uvicorn api.api_main:app --host 0.0.0.0 --port 8000
```

Smoke test:

```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Ignore previous instructions",
    "retrieved_docs": [{"doc_id":"d1","content":"normal context"}],
    "tool_request": {"tool":"get_order","params":{"resource_id":"ORD-001"}},
    "session_context": {"user_id":"user_alice","role":"basic"}
  }' | python3 -m json.tool
```

Post-LLM check (body matches `/analyze` plus `model_output` — use a real completion string in production tests):

```bash
curl -s -X POST http://localhost:8000/analyze-output \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Summarize our refund policy",
    "model_output": "Contact support at leaked@example.com for full card 4111111111111111.",
    "session_context": {"user_id": "u1", "role": "basic"}
  }' | python3 -m json.tool
```

**Dashboard UI:** with the gateway running, start the Streamlit app in a second terminal:

```bash
streamlit run dashboard/app.py
```

It opens at `http://localhost:8501/` and polls the gateway's read-only `/dashboard/*` routes for summary, telemetry tails, and CSVs under `runs/`. Buttons (Run test, Regenerate report, Save target, Delete target) trigger the corresponding `POST`/`DELETE` endpoints listed in *Architecture Snapshot* above.

## Evaluation Scripts

Main scripts in `evaluation/`:
- `run_attack_suite.py`: HTTP attack suite through live gateway
- `attack_failure_analysis.py`: summarize escaped attacks
- `generate_metrics.py`: module-level metrics CSVs
- `measure_latency_breakdown.py`: prompt/rag/fusion stage latency
- `rag_weight_optimization.py`: embedding vs judge weight sweep
- `tune_prompt_threshold.py`: prompt threshold tuning
- `behavior_weight_calibration.py`: behavior signal weight calibration
- `tune_agency_behavior_weights.py`: behavior weight grid search
- `agency_llm_stress_test.py`: real LLM tool-calling stress test
- `ablation_analysis.py`: no-judge/no-deobfuscator/no-behavior analysis
- `run_experiments.py`: batch run of core module tests
- `security_healthcheck.py`: structural/guard health report

Typical run:

```bash
python evaluation/run_attack_suite.py --url http://127.0.0.1:8000 --seed 42
python evaluation/attack_failure_analysis.py
```

## Security Modules

| Module | Main Threats | Key Techniques |
|---|---|---|
| Prompt Guard | jailbreak, obfuscation, instruction override | deobfuscation, adaptive semantic threshold, regex patterns, sanitization |
| RAG Guard | poisoned docs, retrieval hijack, subtle corruption | embedding detector + LLM judge + robust retrieval scoring + context filtering |
| Agency Defense | IDOR/BOLA, sequential probing, bad params, role abuse | object authz, anti-enum, parameter validator, behavior monitor/risk model (tool-call path) |
| Output Guard | PII/secret leakage, unsafe model prose, agent hijack, unknown redirects | pattern + entropy heuristics on `model_output` (only in `analyze_with_output` / batch scripts) |
| Fusion | single-module dilution | weighted sum + critical/elevated max-rule override |

## Config Highlights

`configs/secure_balanced.yaml` controls:
- fusion weights and thresholds
- max-rule override multipliers
- prompt adaptive thresholds
- RAG poison threshold, judge weights, context filter thresholds
- active Ollama model and fallback

## Environment Variables

Common variables:
- `HF_TOKEN`: optional, faster Hugging Face model download
- `OLLAMA_HOST`: default `http://localhost:11434`
- `LLM_JUDGE_MODEL`: optional override for judge model
- `STRICT_SECURITY_STARTUP`: fail app startup on self-check errors when true

## Notes

- `runs/` and `reports/` are generated artifacts.
- First request is slower due to model warmup; `api/startup.py` preloads models at app startup.
- Telemetry (e.g. fusion decisions) is written for observability; dashboard read paths are full-file tail today — when to optimize is documented in `reports/ops_notes.md`.
- External eval CSV column `gateway_miss` counts cases where the gateway *allowed* traffic that should have been blocked/sanitized; it is **not** the same as RAG “attack success rate” (ASR) in `rag_guard/rag_baseline.py`.
- Project layout and setup: `PROJECT_STRUCTURE.md`, `SETUP.md`, `docs/architecture_v1.md`, `docs/design_decisions.md`.
