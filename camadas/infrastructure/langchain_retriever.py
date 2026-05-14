from __future__ import annotations

from langchain_core.vectorstores import VectorStore

from camadas.domain.retriever import DocumentRetriever, RetrievedChunk


class LangChainVectorRetriever(DocumentRetriever):
    """
    Recuperador baseado em qualquer `VectorStore` do LangChain
    (similaridade densa por embedding).
    """

    def __init__(self, vectorstore: VectorStore, *, default_k: int = 5) -> None:
        self._vectorstore = vectorstore
        self._default_k = default_k

    def get_relevant_documents(self, query: str, *, k: int = 5) -> list[RetrievedChunk]:
        effective_k = k or self._default_k
        pairs = self._vectorstore.similarity_search_with_score(query, k=effective_k)
        chunks: list[RetrievedChunk] = []
        for doc, score in pairs:
            chunks.append(
                RetrievedChunk(
                    content=doc.page_content,
                    metadata=dict(doc.metadata or {}),
                    score=float(score),
                )
            )
        return chunks
