# Unified AI Security Gateway

Unified gateway for defending LLM applications against:
- prompt injection / jailbreak attempts
- RAG poisoning and context manipulation
- tool misuse (IDOR, enumeration, param abuse, role misuse)

The system runs three module pipelines in parallel and fuses their scores into a single decision.

## Architecture

```
POST /analyze
    |
    +--> Prompt Guard   (deobfuscate -> normalize -> semantic/pattern -> sanitize)
    +--> RAG Guard      (poison detector -> LLM judge -> retrieval risk -> context filter)
    +--> Agency Defense (authz -> anti-enum -> param validation -> behavior signals)
                  |
                  v
          Fusion Gateway (weighted sum + max-rule override)
                  |
                  v
      allow / sanitize / flag / block
```

## Core Components

- `api/api_main.py`: FastAPI app (`POST /analyze`, `GET /health`)
- `fusion_gateway/engine.py`: module execution + fusion policy
- `prompt_guard/pipeline.py`: full prompt defense pipeline
- `rag_guard/pipeline.py`: hybrid RAG defense pipeline
- `output_agency_defense/*`: agency/tool-call protections
- `schemas/risk_schema.py`: common request/response schema

## Repository Layout

```
unified-ai-security/
├── api/
├── configs/
├── datasets/
├── evaluation/
├── fusion_gateway/
├── output_agency_defense/
├── prompt_guard/
├── rag_guard/
├── schemas/
├── tests/
├── infra/
├── runs/      # generated outputs
└── reports/   # generated reports
```

## Quick Start

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
| Agency Defense | IDOR/BOLA, sequential probing, bad params, role abuse | object authz, anti-enum, parameter validator, behavior monitor/risk model |
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
- Project documentation details live in `PROJECT_STRUCTER.md` and `SETUP.md`.
