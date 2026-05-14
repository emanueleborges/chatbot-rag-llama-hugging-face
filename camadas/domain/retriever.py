from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class RetrievedChunk:
    """Documento ou fragmento retornado pela busca semântica."""

    content: str
    metadata: dict
    score: float | None = None


@runtime_checkable
class DocumentRetriever(Protocol):
    """Porta de recuperação desacoplada do motor vetorial concreto."""

    def get_relevant_documents(self, query: str, *, k: int = 5) -> list[RetrievedChunk]:
        ...


@runtime_checkable
class KnowledgeStore(Protocol):
    """Persistência e indexação de textos na base vetorial."""

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict] | None = None,
    ) -> list[str]:
        ...
