from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class SourceSnippet:
    """Trecho recuperado da base de conhecimento."""

    text: str
    score: float | None = None
    metadata: dict | None = None


@dataclass(frozen=True)
class RAGAnswer:
    """Resposta agregada do RAG (texto + fontes)."""

    text: str
    sources: tuple[SourceSnippet, ...] = ()


@runtime_checkable
class ChatLanguageModel(Protocol):
    """Porta para o modelo de linguagem (local ou remoto)."""

    async def astream_completion(
        self, *, system: str, user_prompt: str
    ) -> AsyncIterator[str]:
        """Fluxo de tokens da resposta."""

    async def acomplete(self, *, system: str, user_prompt: str) -> str:
        """Resposta completa em uma única string."""
