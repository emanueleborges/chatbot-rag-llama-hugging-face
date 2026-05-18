"""Interface com Ollama (LangChain) — com suporte a JSON e health check."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from langchain_ollama import ChatOllama

from backend.config import OLLAMA_MODEL, OLLAMA_TEMPERATURE, OLLAMA_URL


def get_chat_llm(*, temperature: float | None = None) -> ChatOllama:
    return ChatOllama(
        base_url=OLLAMA_URL,
        model=OLLAMA_MODEL,
        temperature=OLLAMA_TEMPERATURE if temperature is None else temperature,
    )


def get_json_llm(*, temperature: float | None = None) -> ChatOllama:
    """
    Retorna um ChatOllama configurado para resposta em JSON.
    Usa `format="json"` para garantir que o modelo retorne JSON válido.
    """
    return ChatOllama(
        base_url=OLLAMA_URL,
        model=OLLAMA_MODEL,
        temperature=OLLAMA_TEMPERATURE if temperature is None else temperature,
        format="json",
    )


def verificar_ollama() -> dict[str, Any]:
    """
    Verifica se o Ollama está acessível e se o modelo está disponível.
    Retorna dict com status, versão e modelo.
    """
    try:
        # Health check básico
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        modelos = [m["name"] for m in data.get("models", [])]
        modelo_disponivel = any(OLLAMA_MODEL in m for m in modelos)

        return {
            "disponivel": True,
            "modelo_configurado": OLLAMA_MODEL,
            "modelo_disponivel": modelo_disponivel,
            "modelos_instalados": modelos,
            "url": OLLAMA_URL,
        }
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError) as e:
        return {
            "disponivel": False,
            "modelo_configurado": OLLAMA_MODEL,
            "erro": str(e),
            "url": OLLAMA_URL,
        }
