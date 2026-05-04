"""Cliente LLM via API compatível OpenAI (Ollama nuvem ou local)."""
from __future__ import annotations

import os
from urllib.parse import urlparse

from openai import OpenAI

_client: OpenAI | None = None


def reset_llm_client() -> None:
    """Limpa cliente em cache (útil em testes ou após mudar OLLAMA_*)."""
    global _client
    _client = None


def _use_placeholder_api_key(base_raw: str, key: str) -> str:
    """
    Ollama local não exige chave real; o SDK OpenAI exige string não vazia.
    Aplica-se a localhost, host.docker.internal e ao serviço Docker `ollama`.
    """
    if key:
        return key
    try:
        host = (urlparse(base_raw).hostname or "").lower()
    except Exception:
        host = ""
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal", "ollama"):
        return "ollama"
    if "127.0.0.1" in base_raw or "localhost" in base_raw:
        return "ollama"
    return key


def get_client() -> OpenAI:
    global _client
    if _client is None:
        base_raw = (os.environ.get("OLLAMA_BASE_URL") or "https://ollama.com/v1").strip().rstrip("/")
        if not base_raw.endswith("/v1"):
            base_raw = f"{base_raw}/v1"
        key = _use_placeholder_api_key(base_raw, (os.environ.get("OLLAMA_API_KEY") or "").strip())
        if not key:
            raise RuntimeError(
                "Defina OLLAMA_API_KEY no .env (nuvem: ollama.com/settings/keys) ou use "
                "OLLAMA_BASE_URL=http://127.0.0.1:11434/v1 (ou http://ollama:11434/v1 no Docker Compose) "
                "com Ollama a correr."
            )
        _client = OpenAI(api_key=key, base_url=base_raw)
    return _client


def chat_completion(messages: list[dict], model: str | None = None) -> str:
    m = (model or os.environ.get("OLLAMA_MODEL") or "gpt-oss:20b").strip() or "gpt-oss:20b"
    client = get_client()
    response = client.chat.completions.create(model=m, messages=messages)
    return (response.choices[0].message.content or "").strip()
