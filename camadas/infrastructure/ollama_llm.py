from __future__ import annotations

import os
from collections.abc import AsyncIterator

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from camadas.infrastructure.llm_transport_errors import (
    friendly_llm_unreachable_message,
    is_llm_transport_error,
)


class OllamaLangChainModel:
    """Adaptador Ollama compatível com a porta `ChatLanguageModel`."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
    ) -> None:
        self._base_url = base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self._chat = ChatOllama(
            model=model or os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            base_url=self._base_url,
            temperature=temperature,
        )

    async def astream_completion(
        self, *, system: str, user_prompt: str
    ) -> AsyncIterator[str]:
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user_prompt),
        ]
        try:
            async for chunk in self._chat.astream(messages):
                content = getattr(chunk, "content", None) or ""
                if content:
                    yield content
        except BaseException as exc:
            if is_llm_transport_error(exc):
                yield friendly_llm_unreachable_message(
                    backend="Ollama",
                    endpoint=self._base_url,
                )
                return
            raise

    async def acomplete(self, *, system: str, user_prompt: str) -> str:
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user_prompt),
        ]
        try:
            result = await self._chat.ainvoke(messages)
            return getattr(result, "content", "") or ""
        except BaseException as exc:
            if is_llm_transport_error(exc):
                return friendly_llm_unreachable_message(
                    backend="Ollama",
                    endpoint=self._base_url,
                )
            raise
