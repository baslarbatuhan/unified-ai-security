<div align="center">

# 🛡️ Unified AI Security Gateway

**A multi-layer defense framework for Large Language Model applications.**

One gateway that screens prompts, retrieved context, tool calls, and model output —
fuses the evidence into a single verdict, and (optionally) *enforces* it before the
request ever reaches your chatbot.

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/deploy-Docker%20Compose-2496ED?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/tests-608%20passing-brightgreen)
![Status](https://img.shields.io/badge/status-capstone%20project-blueviolet)

</div>

---

## ✨ What it defends against

| Threat surface | Module | Techniques |
|---|---|---|
| 🧬 **Prompt injection / jailbreak** | Prompt Guard | deobfuscation → NFKC normalize → BGE-M3 semantic + regex → sanitize |
| 📄 **RAG poisoning / context hijack** | RAG Guard | embedding detector + **LLM judge** + chunked analysis + context filter |
| 🔧 **Tool misuse** (IDOR/BOLA, enumeration, param abuse, role misuse) | Agency Defense | object authz + **stateful** anti-enum + param validator + behavior signals |
| 🕵️ **Output-layer leakage** (PII, secrets, unsafe prose, redirects) | Output Guard | regex + entropy on the model completion (`/analyze-output`) |
| 🧠 **Single-module dilution** | Fusion Gateway | weighted sum + critical/elevated **max-rule override** |

The system runs the input-side modules **in parallel** and fuses their scores into one
decision (`allow` / `sanitize` / `flag` / `block`). A separate post-LLM path adds output
analysis. For external targets it supports two enforcement modes:

| Mode | Behaviour |
|------|-----------|
| **Passthrough** *(default)* | Every prompt reaches the chatbot; the gateway scores and logs only (observability / RLHF comparison). |
| **🔥 Firewall** (`--firewall` or `target.policy.mode: firewall`) | Gateway runs **first**; a `block` verdict **skips the adapter — the prompt never reaches the target**. |

---

## 📑 Table of Contents

- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [External Evaluation & Firewall](#-external-evaluation--firewall)
- [Dashboard](#-dashboard)
- [Configuration](#-configuration)
- [Tests](#-tests)
- [Project Layout](#-project-layout)
- [Documentation](#-documentation)

---

## 🏗 Architecture

**Pre-LLM (input screening)** — the gateway scores the prompt/context; it does *not* call the target LLM.

```
POST /analyze
    │
    ├─▶ Prompt Guard   (deobfuscate → normalize → semantic/pattern → sanitize)
    ├─▶ RAG Guard      (poison detector → LLM judge → retrieval risk → context filter)
    └─▶ Agency Defense (tool/authz → anti-enum → param validation → behavior)
                  │
                  ▼
          Fusion Gateway (weighted sum + max-rule override)
                  │
                  ▼
      allow / sanitize / flag / block      (output_score = 0.0 on this path)
```

**Post-LLM (output screening)** — after the *client* calls its own LLM, it posts the completion:

```
POST /analyze-output  (requires model_output)
    │
    ├─▶ same three input-side modules (stateless re-check)
    └─▶ Output Guard   (PII, API-key-like tokens, unsafe text, injection, off-allowlist URLs)
                  │
                  ▼
          Fusion (four modules; output_score reflects output_guard)
```

**External evaluation per case (gateway-first, then enforce):**

```
load_suite() ─▶ for each AttackCase:
    gateway.analyze(prompt)          # verdict FIRST
    firewall_policy.decide(...)      # forward or drop (per target policy)
    adapter.send(prompt)             # only if forwarded
    └─▶ runs/<run_id>/results.csv
```

---

## 🚀 Quick Start

### Option A — Docker (recommended)

```bash
git clone https://github.com/baslarbatuhan/unified-ai-security.git
cd unified-ai-security/infra
cp .env.example .env          # optional: HF_TOKEN, MY_OPENWEBUI_KEY, …
docker compose up -d --build
```

| Service | URL |
|---------|-----|
| 🛡️ Gateway (FastAPI) | http://localhost:8000 |
| 📊 Dashboard (Streamlit) | http://localhost:8501 |
| 💬 Open WebUI (demo target) | http://localhost:3000 |
| 🗂️ ChromaDB | localhost:8001 |
| 🦙 Ollama | localhost:11435 |

> First boot: gateway warm-up takes ~60–90 s (`UAIS_WARM_UP_PIPELINES=1` in compose).

Pull the Ollama models once:

```bash
docker exec uais-ollama ollama pull qwen2.5:7b   # RAG judge
docker exec uais-ollama ollama pull qwen2.5:3b   # Open WebUI chat target
```

**GPU** (NVIDIA + Container Toolkit):

```bash
docker compose -f docker-compose.yml \
               -f docker-compose.override.yml \
               -f docker-compose.gpu.yml up -d --build
```

<details>
<summary><b>Open WebUI as a live target (API + web)</b></summary>

API target — Open WebUI → Settings → Account → API key, then add to `infra/.env`:

```bash
MY_OPENWEBUI_KEY=sk-...
```

…or paste it via Dashboard → **Targets** vault (`runs/.secrets.yaml`, never committed).

Web target (Playwright) — export the login state once:

```bash
python scripts/export_webui_state.py --url http://localhost:3000 --out runs/open_webui_state.json
```
</details>

### Option B — Local Python venv

Requires Python **3.10+** (the Docker image uses 3.11).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
playwright install chromium      # only for type=web targets

uvicorn api.api_main:app --host 0.0.0.0 --port 8000   # gateway
streamlit run dashboard/app.py                         # dashboard (2nd terminal)
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

---

## 🔥 External Evaluation & Firewall

Run an attack suite against a registered target (gateway must be up). Targets live in
`external_eval/targets.yaml` and can be edited from the dashboard.

```bash
# Observability (default): every prompt reaches the chatbot
python external_eval/run_external_eval.py --target open_webui_api --suite prompt_injection --max-attacks 10

# Firewall: a `block` verdict means the prompt is NEVER sent to the target
python external_eval/run_external_eval.py --target open_webui_api --suite prompt_injection --firewall
```

Or set `policy.mode: firewall` on the target, or use Dashboard → **Run test** → **Firewall** toggle.

- **Suites:** `prompt_injection`, `rag_poisoning`, `agency_social`, `all` (+ single-prompt mode in the dashboard).
- **Adapters:** `api` (REST), `web` (Playwright), `tools_local` (real Python tool dispatch), `mock`.
- **Auth:** six variants — `none`, `bearer`, `header`, `query`, `basic`, `cookie` — secrets resolved via env → file-locked vault → `${VAR}` interpolation.
- **CSV trace:** `gateway_decision`, `gateway_miss`, plus firewall columns `mode`, `firewall_action`, `forwarded_to_target`, `adapter_state`.

> `forwarded_to_target = 0` + `adapter_state = blocked_by_gateway_pre` = the firewall stopped the prompt before the chatbot was ever called.

---

## 📊 Dashboard

A nine-page Streamlit app (`dashboard/app.py`) that talks to the gateway over read-only HTTP:

| Page | What it shows |
|---|---|
| **Admin** | `/health` checks, circuit-breaker & rate-limiter state, raw telemetry tail |
| **Home** | composite security score, KPI strip, attack-class distribution |
| **Targets** | CRUD + 9 provider presets, 🔐 secret vault, test-connection diagnostics |
| **Run test** | single-shot `/analyze`(`-output`) **or** suite runs with **firewall toggle** + coupled weight sliders |
| **Live monitor** | per-module risk/latency tail, global vs. per-run scope |
| **Results** | per-run analytics, **firewall enforcement strip**, FN/FP inspector, decision-trace drill-down |
| **Logs** | per-decision explainability (which signal fired and why) |
| **Reports** | view / regenerate / download Markdown + PDF reports |
| **Compare runs** | side-by-side diff incl. **passthrough-vs-firewall** enforcement section |

---

## ⚙️ Configuration

| File | Purpose |
|---|---|
| `configs/secure_balanced.yaml` | fusion weights, thresholds, RAG judge settings |
| `configs/timeout_config.yaml` | per-module budgets + fail-closed policies |
| `configs/service_limits.yaml` | rate limits & eval parallelism |
| `configs/security_score_weights.yaml` | Home-page composite-score weights |

**Key environment variables**

| Variable | Where | Purpose |
|---|---|---|
| `HF_TOKEN` | `.env` | Faster Hugging Face model downloads |
| `UAIS_WARM_UP_PIPELINES=1` | compose (gateway) | Eager-load BGE-M3 + pipelines at startup |
| `OLLAMA_HOST` | gateway | Ollama endpoint (compose: `http://ollama:11434`) |
| `OLLAMA_KEEP_ALIVE` | compose (ollama) | Keep models resident (default `30m`) so the first RAG case skips cold-start |
| `MY_OPENWEBUI_KEY` | `.env` or vault | Bearer token for the `open_webui_api` target |
| `GATEWAY_URL` | dashboard | Gateway base URL (compose: `http://gateway:8000`) |

Target credentials use `auth.token_env: <NAME>` in `targets.yaml`; values come from the
environment or the file-locked vault `runs/.secrets.yaml` (never committed).

---

## 🧪 Tests

```bash
pytest tests/ -q          # 609 collected · 608 pass · 1 skip
```

---

## 🗂 Project Layout

```
unified-ai-security/
├── api/                    # FastAPI gateway, routes, middleware
├── dashboard/              # Streamlit UI (app.py + pages/ + lib/)
├── fusion_gateway/         # parallel module execution, fusion, firewall_policy
├── prompt_guard/           # deobfuscator, semantic eval, patterns, sanitizer
├── rag_guard/              # poison detector, LLM judge, retrieval risk, filter
├── output_agency_defense/  # object authz, anti-enum, param validator
├── output_guard/           # PII / secret / unsafe-text scanners
├── external_eval/          # adapters, targets.yaml, run_external_eval.py
├── tools/                  # local tool registry (tools_local target)
├── schemas/                # request/response + target schemas
├── configs/ · utils/ · evaluation/ · scripts/ · datasets/
├── tests/                  # 609 tests
├── infra/                  # docker-compose*.yml, .env.example
├── docs/                   # architecture, threat models, design decisions
├── runs/ · reports/        # generated artefacts (gitignored)
```

---

## 📚 Documentation

- **[SETUP.md](SETUP.md)** — install, Docker, Open WebUI, troubleshooting
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** — file map, data flow, CSV columns
- **[docs/architecture_v1.md](docs/architecture_v1.md)** — HTTP routes, adapter axis, enforcement
- **[docs/glossary.md](docs/glossary.md)** — `gateway_miss` vs ASR, key terms

---

<div align="center">

*Capstone project · Department of Computer Engineering, İstanbul Kültür University.*
*`gateway_miss` is a **protector** metric (expected block/sanitize but gateway allowed) — not a RAG retrieval attack-success rate.*

</div>
