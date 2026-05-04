# RAG Chat — FastAPI + Chroma + Sentence-Transformers (imagem CPU ~2GB+ após build)
FROM python:3.12-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# CI: docker build --target ci .  (não é estágio predefinido do compose; compose usa o último estágio)
FROM base AS ci
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt
RUN pytest -q

FROM base AS production

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://127.0.0.1:8000/api/health > /dev/null || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
