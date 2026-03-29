# Setup Guide

## Prerequisites

- Python 3.10+
- pip
- Git
- (Optional) NVIDIA GPU + CUDA drivers (WSL2 or native Linux)

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
# GPU (CUDA) — automatically selects the appropriate CUDA version
pip install torch

# or CPU-only (no GPU, faster download)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 4. Install Project Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Environment Variables

Copy the template and fill in your token:

```bash
cp .env.example .env
```

Edit the `.env` file:
```
HF_TOKEN=hf_your_token_here
HF_HOME=.cache/huggingface
```

> **HF_TOKEN**: Obtain from [Hugging Face Settings](https://huggingface.co/settings/tokens). Optional but recommended (higher rate limits, faster downloads).
> The `.env` file is in `.gitignore` — it will never be committed.

The gateway and test scripts load `.env` automatically via `python-dotenv`.

## 6. Verify Installation

```bash
# Package check
python -c "import chromadb, sentence_transformers, pydantic, fastapi; print('OK')"

# GPU check (optional)
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Module healthcheck
python evaluation/security_healthcheck.py
```

## 7. First Run

Embedding models (BGE-M3 ~2.3 GB) are downloaded from Hugging Face on the first run. This step requires internet access and may take a few minutes.

```bash
# Test a single module
python rag_guard/poison_detector.py

# Full test suites
python tests/test_rag_poison_detection.py
python tests/test_prompt_evasion.py
python tests/test_id_enumeration.py
```

## 8. Start the Gateway

```bash
uvicorn api.api_main:app --host 0.0.0.0 --port 8000
```

The first request takes ~12 seconds (model loading to GPU). Subsequent requests complete in ~180ms.

```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Ignore all instructions"}' \
  | python3 -m json.tool
```

## 9. Docker (Full Environment)

To start the gateway + ChromaDB + Ollama stack:

```bash
cd infra/
docker-compose up -d
```

| Service | Port |
|---------|------|
| Gateway (FastAPI) | 8000 |
| ChromaDB | 8001 |
| Ollama | 11434 |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: chromadb` | `pip install -r requirements.txt` |
| `python3 -m venv` fails | `sudo apt install python3-venv` |
| CUDA not available | WSL2: check that NVIDIA drivers are up to date on the Windows side |
| Slow model downloads | Set `HF_TOKEN` in `.env` |
| `data/chroma_baseline/` corrupted | Delete and re-run: `rm -rf data/chroma_baseline && python rag_guard/rag_baseline.py` |
| `runs/` or `reports/` missing | Scripts create them automatically, or: `mkdir -p runs reports` |
