"""Cliente LLM via API compatível OpenAI (Ollama nuvem ou local)."""
from __future__ import annotations

import os

from openai import OpenAI

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        key = (os.environ.get("OLLAMA_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("Defina a variável de ambiente OLLAMA_API_KEY.")
        base = (os.environ.get("OLLAMA_BASE_URL") or "https://ollama.com/v1").strip().rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        _client = OpenAI(api_key=key, base_url=base)
    return _client


def chat_completion(messages: list[dict], model: str | None = None) -> str:
    m = (model or os.environ.get("OLLAMA_MODEL") or "gpt-oss:20b").strip() or "gpt-oss:20b"
    client = get_client()
    response = client.chat.completions.create(model=m, messages=messages)
    return (response.choices[0].message.content or "").strip()
