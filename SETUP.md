# Setup Guide

## Prerequisites

- Python 3.10+
- pip
- Git
- Ollama (LLM inference — qwen2.5:7b, llama3.1:8b, gemma2:2b)
- (Optional) NVIDIA GPU + CUDA drivers for Ollama acceleration
- (Optional) Docker + Docker Compose for containerized deployment

---

## 1. Clone the Repository

```bash
git clone https://github.com/<user>/unified-ai-security.git
cd unified-ai-security
```

## 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

> If `python3 -m venv` fails: `sudo apt install python3-venv` (Ubuntu/Debian)

## 3. Install PyTorch

```bash
# CPU-only (recommended — GPU is reserved for Ollama)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# or GPU (CUDA) if you have enough VRAM for both embeddings and LLM
pip install torch
```

## 4. Install Project Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b
ollama pull llama3.1:8b     # fallback
ollama pull gemma2:2b       # second fallback
```

## 6. Environment Variables

```bash
cp .env.example .env
```

Edit `.env`:
```
HF_TOKEN=hf_your_token_here
HF_HOME=.cache/huggingface
EMBEDDING_DEVICE=cpu
```

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | Hugging Face token for faster model downloads (get from https://huggingface.co/settings/tokens) |
| `HF_HOME` | Local cache directory for HF models |
| `EMBEDDING_DEVICE` | `cpu` = embeddings on CPU (recommended), `cuda` = embeddings on GPU |

> `.env` is in `.gitignore` — it will never be committed.

## 7. Verify Installation

```bash
# Package check
python -c "import chromadb, sentence_transformers, pydantic, fastapi, docker; print('OK')"

# GPU check
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Ollama check
curl -s http://localhost:11434/api/tags | python3 -m json.tool

# Module healthcheck
python evaluation/security_healthcheck.py
```

## 8. First Run

Embedding models (BAAI/bge-m3, ~2.3 GB) are downloaded on the first run.

```bash
# Test individual modules
python rag_guard/poison_detector.py
python prompt_guard/semantic_evaluator_v1.py
python output_agency_defense/tool_call_simulator.py

# Full test suites
python tests/test_rag_poison_detection.py
python tests/test_prompt_evasion.py
python tests/test_id_enumeration.py
python tests/test_agency_attack_scenarios.py
python tests/test_behavior_monitor.py
```

## 9. Start the Gateway (Local)

```bash
uvicorn api.api_main:app --host 0.0.0.0 --port 8000
```

First request takes ~10s (model loading). Subsequent requests: ~100-200ms.

```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Ignore all instructions",
    "session_context": {"user_id": "user_alice", "role": "basic"}
  }' | python3 -m json.tool
```

## 10. Docker (Full Environment)

```bash
cd infra/
docker compose up --build -d
```

| Service | Host Port | Container Port |
|---------|-----------|----------------|
| Gateway (FastAPI) | 8000 | 8000 |
| ChromaDB | 8001 | 8000 |
| Ollama | 11435 | 11434 |
| Sandbox | — | isolated |

GPU allocation in Docker:
- **Ollama**: Full GPU access (`runtime: nvidia`)
- **Gateway**: CPU only (`EMBEDDING_DEVICE=cpu`)
- **Sandbox**: No GPU, no network, read-only filesystem

## 11. Generate Evaluation Outputs

```bash
# Requires running gateway + Ollama
python evaluation/run_attack_suite.py --http
python evaluation/generate_metrics.py
python evaluation/fusion_threshold_optimization.py
```

Outputs in `runs/`:
- `gateway_attack_results.csv` — 147 attacks via /analyze
- `prompt_metrics.csv` — Prompt guard evaluation
- `rag_metrics.csv` — RAG guard hybrid pipeline
- `agency_metrics.csv` — Agency with LLM tool calling (30/30)
- `latency_metrics.csv` — Per-module timing
- `fusion_threshold_analysis.csv` — 1440 threshold combinations

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: chromadb` | `pip install -r requirements.txt` |
| `python3 -m venv` fails | `sudo apt install python3-venv` |
| CUDA not available | WSL2: update NVIDIA drivers on Windows side |
| Slow model downloads | Set `HF_TOKEN` in `.env` |
| Ollama not responding | `ollama serve` or check `systemctl status ollama` |
| Port 8000 in use | `sudo fuser -k 8000/tcp` or stop Docker: `docker compose down` |
| `data/chroma_baseline/` corrupted | `rm -rf data/chroma_baseline && python rag_guard/rag_baseline.py` |
| Docker GPU issues | Check `nvidia-smi`, ensure `nvidia-container-toolkit` installed |
