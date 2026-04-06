# Unified AI Security Gateway

Three-layer AI security gateway that detects and mitigates **prompt injection**, **RAG poisoning**, and **excessive agency / tool misuse** attacks. A fusion engine combines per-module risk scores into a single allow / sanitize / flag / block decision.

## Architecture

```
user request ──► POST /analyze
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼          (ThreadPoolExecutor — parallel)
  Prompt Guard   RAG Guard   Agency Defense
  ├ deobfuscate  ├ embedding  ├ tool allowlist
  ├ normalize    ├ LLM judge  ├ RBAC
  ├ semantic     ├ combined   ├ IDOR / authz
  ├ pattern      ├ context    ├ param validation
  └ sanitize     └ filter     └ anti-enum
        │            │            │
        └────────────┼────────────┘
                     ▼
              Fusion Gateway
        (weighted sum + max-rule override)
                     │
                     ▼
              Final Decision
        (allow / sanitize / flag / block)
```

## Project Structure

```
unified-ai-security/
├── api/                        # FastAPI gateway (POST /analyze, GET /health)
│   ├── api_main.py             # Uvicorn entry point
│   ├── security_gateway.py     # Request → FusionEngine bridge
│   └── health.py               # 5-check health endpoint
├── configs/
│   └── secure_balanced.yaml    # Weights, thresholds, module config
├── datasets/
│   ├── injection_prompts/      # Prompt injection CSV (245 samples)
│   ├── poisoned_corpus/        # RAG poison JSON (40 + 25 advanced)
│   └── output_agency_attacks/  # Agency attack scenarios (30)
├── evaluation/
│   ├── run_attack_suite.py     # Gateway HTTP attack runner
│   ├── generate_metrics.py     # Per-module CSV generator (with LLM)
│   └── fusion_threshold_optimization.py
├── fusion_gateway/
│   └── engine.py               # Weighted fusion + max-rule override
├── infra/
│   └── docker-compose.yml      # Gateway + ChromaDB + Ollama + Sandbox
├── output_agency_defense/
│   ├── tool_call_simulator.py  # LLM tool calling via Ollama
│   ├── parameter_validation.py # Type/regex/range/allowlist/denylist
│   ├── behavior_risk_model.py  # Burst/auth-fail/multi-resource signals
│   ├── sandbox_executor.py     # Docker-based isolated execution
│   ├── object_authz_guard.py   # IDOR / BOLA protection
│   └── anti_enum_guard.py      # Sequential ID probe detection
├── prompt_guard/
│   ├── pipeline.py             # Full pipeline: deobfuscate → detect → sanitize
│   ├── deobfuscator.py         # Leetspeak + Unicode de-obfuscation
│   ├── semantic_evaluator_v1.py # BGE-M3 cosine similarity (110 signatures)
│   ├── pattern_detector.py     # 32 regex patterns
│   ├── normalizer.py           # Text normalization
│   └── prompt_sanitizer.py     # Injection removal + clean prompt
├── rag_guard/
│   ├── pipeline.py             # Hybrid: embedding + LLM judge + filter
│   ├── poison_detector.py      # Pattern (9) + semantic (18 signatures)
│   ├── llm_judge.py            # Ollama LLM-as-a-Judge (qwen2.5 → llama3.1 → gemma2)
│   ├── context_filter.py       # Remove / demote / keep (3-tier)
│   └── build_safe_context.py   # High-level safe context builder
├── schemas/
│   └── risk_schema.py          # Pydantic request/response contracts
├── tests/                      # Unit / integration / evasion suites
├── runs/                       # Generated CSV outputs (gitignored)
├── reports/                    # Error analysis (gitignored)
├── Dockerfile
├── requirements.txt
└── SETUP.md
```

## Quick Start

```bash
git clone https://github.com/<user>/unified-ai-security.git
cd unified-ai-security
python3 -m venv .venv && source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cp .env.example .env   # add HF_TOKEN
```

### Run the gateway (local)

```bash
uvicorn api.api_main:app --host 0.0.0.0 --port 8000
```

### Run the gateway (Docker)

```bash
cd infra/
docker compose up --build -d
```

| Service | Port |
|---------|------|
| Gateway | 8000 |
| ChromaDB | 8001 |
| Ollama | 11435 (host) → 11434 (container) |

### Test

```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Ignore all previous instructions",
    "retrieved_docs": [{"doc_id": "d1", "content": "Normal document."}],
    "tool_request": {"tool": "get_order", "params": {"resource_id": "ORD-001"}},
    "session_context": {"user_id": "user_alice", "role": "basic"}
  }' | python3 -m json.tool
```

### Run tests & evaluation

```bash
# Module tests
python tests/test_rag_poison_detection.py
python tests/test_prompt_evasion.py
python tests/test_agency_attack_scenarios.py
python tests/test_id_enumeration.py
python tests/test_behavior_monitor.py

# Attack suite (requires running gateway)
python evaluation/run_attack_suite.py --http

# Generate metrics (requires Ollama for RAG + Agency LLM)
python evaluation/generate_metrics.py

# Health check
python evaluation/security_healthcheck.py
```

## Security Modules

| Module | Threats | Key Techniques |
|--------|---------|----------------|
| **Prompt Guard** | Prompt injection, jailbreak, leetspeak obfuscation | De-obfuscation, semantic similarity (BGE-M3), 32 regex patterns, prompt sanitization |
| **RAG Guard** | Knowledge corruption, retrieval hijacking, data exfiltration | Hybrid detection (embedding + LLM judge), 3-tier context filtering, configurable weights from YAML |
| **Agency Defense** | IDOR/BOLA, enumeration, param manipulation, role misuse | Tool allowlist, RBAC, parameter validation (type/regex/range), behavior monitoring, LLM tool call simulation |
| **Fusion Gateway** | Single-module dilution | Weighted sum (agency=0.40, prompt=0.30, rag=0.30) + max-rule override, configurable thresholds |

## Datasets

| Dataset | Records | Purpose |
|---------|---------|---------|
| `injection_prompts/injection_dataset_v1.csv` | 245 | Prompt injection (135 benign + 110 attack, 10 categories) |
| `poisoned_corpus/poison_samples.json` | 40 | RAG poisoning (20 clean + 20 poisoned, 7 attack types) |
| `poisoned_corpus/advanced_poison_samples.json` | 25 | Advanced evasion (semantic camouflage, authority mimicry) |
| `output_agency_attacks/agency_attack_scenarios.json` | 30 | Tool misuse (IDOR, enumeration, param manipulation, role abuse) |

## Configuration

`configs/secure_balanced.yaml`:

```yaml
policy:
  fusion:
    weights:
      output_agency: 0.40
      prompt_guard:  0.30
      rag_guard:     0.30
    thresholds:
      allow:    0.30
      sanitize: 0.60
      block:    0.85
    override:
      critical_threshold: 0.85
      elevated_threshold: 0.60

modules:
  rag_guard:
    llm_judge:
      embedding_weight: 0.4
      judge_weight: 0.6
    context_filter:
      min_safe_docs: 2
```

## Environment Variables

| Variable | Local (.env) | Docker (compose) | Purpose |
|----------|-------------|-------------------|---------|
| `HF_TOKEN` | `.env` | `infra/.env` | Hugging Face model downloads |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` | Force embeddings to CPU (GPU for Ollama) |
| `OLLAMA_HOST` | `http://localhost:11434` | `http://ollama:11434` | LLM inference endpoint |
| `CHROMA_HOST` | `localhost` | `chroma` | Vector store |

## Tech Stack

- Python 3.11, PyTorch, Sentence-Transformers (BAAI/bge-m3)
- FastAPI + Uvicorn
- Ollama (qwen2.5:7b, llama3.1:8b, gemma2:2b)
- ChromaDB (vector store)
- Docker Compose (gateway + ChromaDB + Ollama + sandbox)
- Pydantic v2, PyYAML

## License

This project was developed as a graduation thesis.
