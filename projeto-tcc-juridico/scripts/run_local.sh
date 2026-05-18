#!/usr/bin/env bash
# Arranque local (sem Docker) — projeto-tcc-juridico
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -d "$ROOT/.venv" ]]; then
  PY="$ROOT/.venv/bin/python"
  PIP="$ROOT/.venv/bin/pip"
elif [[ -d "$ROOT/venv" ]]; then
  PY="$ROOT/venv/bin/python"
  PIP="$ROOT/venv/bin/pip"
else
  PY=python3
  PIP=pip3
fi

if [[ ! -f "$ROOT/.env" ]]; then
  if [[ -f "$ROOT/../.env" ]]; then
    echo "Usando $ROOT/../.env"
  elif [[ -f "$ROOT/.env.example" ]]; then
    cp "$ROOT/.env.example" "$ROOT/.env"
    echo "Criado $ROOT/.env a partir de .env.example — ajuste OLLAMA_MODEL se necessário."
  fi
fi

case "${1:-gradio}" in
  install)
    "$PIP" install -r backend/requirements.txt
    ;;
  api)
    echo "API: http://127.0.0.1:${JURIDICO_API_PORT:-8011}"
    exec "$PY" -m backend.app
    ;;
  gradio)
    echo "Gradio: http://127.0.0.1:${GRADIO_PORT:-7861}"
    exec "$PY" frontend/gradio_app.py
    ;;
  *)
    echo "Uso: $0 [install|api|gradio]"
    exit 1
    ;;
esac
