from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda

from camadas.domain.rerank import DocumentReranker
from camadas.domain.retriever import DocumentRetriever, RetrievedChunk


def build_retrieve_and_rerank_runnable(
    retriever: DocumentRetriever,
    reranker: DocumentReranker,
    *,
    fetch_k: int,
    final_k: int,
) -> Runnable:
    """
    Cadeia LangChain (Runnable) que combina busca vetorial e re-ranking contextual.
    Entrada: {"standalone_query": str}. Saída: lista de `RetrievedChunk`.
    """

    def _run(payload: dict[str, Any]) -> list[RetrievedChunk]:
        query = str(payload["standalone_query"])
        raw = retriever.get_relevant_documents(query, k=fetch_k)
        return reranker.rerank(query, raw, top_k=final_k)

    return RunnableLambda(_run)
