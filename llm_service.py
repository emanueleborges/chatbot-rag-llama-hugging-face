"""Cliente LLM via API compatível OpenAI (Ollama nuvem ou local)."""
from __future__ import annotations

import os

from openai import OpenAI

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        base_raw = (os.environ.get("OLLAMA_BASE_URL") or "https://ollama.com/v1").strip().rstrip("/")
        if not base_raw.endswith("/v1"):
            base_raw = f"{base_raw}/v1"
        key = (os.environ.get("OLLAMA_API_KEY") or "").strip()
        # API local (sem conta na nuvem): o SDK exige string; o Ollama local ignora.
        if not key and ("127.0.0.1" in base_raw or "localhost" in base_raw):
            key = "ollama"
        if not key:
            raise RuntimeError(
                "Defina OLLAMA_API_KEY no .env (nuvem: ollama.com/settings/keys) ou use "
                "OLLAMA_BASE_URL=http://127.0.0.1:11434/v1 com Ollama instalado no PC."
            )
        _client = OpenAI(api_key=key, base_url=base_raw)
    return _client


def chat_completion(messages: list[dict], model: str | None = None) -> str:
    m = (model or os.environ.get("OLLAMA_MODEL") or "gpt-oss:20b").strip() or "gpt-oss:20b"
    client = get_client()
    response = client.chat.completions.create(model=m, messages=messages)
    return (response.choices[0].message.content or "").strip()
