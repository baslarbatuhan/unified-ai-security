FROM python:3.11-slim

WORKDIR /app

# cpu = smaller image (e.g. sandbox service in compose)
# cuda = default for gateway when built via infra/docker-compose.yml (needs --gpus at run)
ARG TORCH_DEVICE=cuda

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN if [ "$TORCH_DEVICE" = "cuda" ]; then \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128; \
    else \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    fi && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Environment
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=http://ollama:11434
ENV CHROMA_HOST=chroma
ENV CHROMA_PORT=8000
ENV EMBEDDING_DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "api.api_main:app", "--host", "0.0.0.0", "--port", "8000"]
