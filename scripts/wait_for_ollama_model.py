"""Espera pelo Ollama HTTP e garante que o modelo existe (pull se necessário). CI e local."""
from __future__ import annotations

import os
import sys
import time

import httpx

HOST = os.environ.get("OLLAMA_PULL_HOST", "http://127.0.0.1:11434").rstrip("/")
MODEL = (os.environ.get("OLLAMA_MODEL") or "llama3.2:1b").strip()


def _ready() -> bool:
    try:
        r = httpx.get(f"{HOST}/api/tags", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


def _model_exists() -> bool:
    try:
        r = httpx.post(f"{HOST}/api/show", json={"model": MODEL}, timeout=60.0)
        return r.status_code == 200
    except Exception:
        return False


def _pull() -> None:
    r = httpx.post(
        f"{HOST}/api/pull",
        json={"model": MODEL, "stream": False},
        timeout=httpx.Timeout(900.0),
    )
    r.raise_for_status()


def main() -> int:
    print(f"A aguardar Ollama em {HOST} ...")
    for _ in range(90):
        if _ready():
            break
        time.sleep(2)
    else:
        print("Ollama não respondeu.", file=sys.stderr)
        return 1
    if _model_exists():
        print(f"Modelo {MODEL} já disponível.")
        return 0
    print(f"A fazer pull de {MODEL} (pode demorar) ...")
    try:
        _pull()
    except Exception as e:
        print(f"Pull falhou: {e}", file=sys.stderr)
        return 1
    if not _model_exists():
        print("Modelo não apareceu após pull.", file=sys.stderr)
        return 1
    print("Pull concluído.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
