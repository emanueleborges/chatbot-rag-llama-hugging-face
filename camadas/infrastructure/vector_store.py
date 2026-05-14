from __future__ import annotations

import os
from typing import Sequence

from langchain_community.vectorstores import Chroma

from camadas.domain.retriever import KnowledgeStore


def build_chroma_vector_store(
    embedding_function,
    *,
    persist_directory: str | None = None,
    collection_name: str | None = None,
) -> Chroma:
    persist_directory = persist_directory or os.getenv(
        "CHROMA_PERSIST_DIR",
        "./chroma_db_rag",
    )
    collection_name = collection_name or os.getenv("CHROMA_COLLECTION", "rag_langchain")
    return Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )


class ChromaKnowledgeStore(KnowledgeStore):
    """Adaptador de ingestão sobre `Chroma` do LangChain."""

    def __init__(self, store: Chroma) -> None:
        self._store = store

    @property
    def vectorstore(self) -> Chroma:
        return self._store

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict] | None = None,
    ) -> list[str]:
        texts_list = list(texts)
        metas = list(metadatas) if metadatas is not None else None
        ids = self._store.add_texts(texts=texts_list, metadatas=metas)
        if hasattr(self._store, "persist"):
            self._store.persist()
        return ids
