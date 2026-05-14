from __future__ import annotations

from typing import Protocol, runtime_checkable

from camadas.domain.retriever import RetrievedChunk


@runtime_checkable
class DocumentReranker(Protocol):
    def rerank(
        self,
        query: str,
        documents: list[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        ...
