FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ backend/
COPY frontend/ frontend/
COPY data/ data/

ENV PYTHONPATH=/app
ENV JURIDICO_API_PORT=8011
ENV GRADIO_PORT=7861
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV HF_HOME=/app/.cache/huggingface
