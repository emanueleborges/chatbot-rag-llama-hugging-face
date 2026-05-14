from __future__ import annotations

import os

from sentence_transformers import CrossEncoder

from camadas.domain.rerank import DocumentReranker
from camadas.domain.retriever import RetrievedChunk


class IdentityDocumentReranker(DocumentReranker):
    """Sem re-ranking: apenas trunca à lista top_k na ordem original."""

    def rerank(
        self,
        query: str,
        documents: list[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        _ = query
        return documents[: max(1, top_k)]


class CrossEncoderDocumentReranker(DocumentReranker):
    """Re-ranking com Cross-Encoder (relevância query-documento)."""

    def __init__(self, *, model_name: str | None = None) -> None:
        self._model_name = model_name or os.getenv(
            "RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        self._model: CrossEncoder | None = None

    def _ensure_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not documents:
            return []
        model = self._ensure_model()
        pairs = [(query, d.content) for d in documents]
        scores = model.predict(pairs)
        ranked = sorted(
            zip(documents, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        trimmed = ranked[: max(1, top_k)]
        return [
            RetrievedChunk(
                content=doc.content,
                metadata=dict(doc.metadata),
                score=float(score),
            )
            for doc, score in trimmed
        ]
