"""ChromaDB + embeddings (LangChain) para documentos jurídicos."""
from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import (
    CHROMA_DIR,
    COLLECTION_DOCS,
    COLLECTION_REFINE,
    EMBEDDING_MODEL,
    RAG_TOP_K,
)
from backend.ingest import (
    documents_from_path,
    iter_knowledge_files,
    load_exemplo_json,
    text_to_documents,
)

_embeddings: HuggingFaceEmbeddings | None = None


def _collection_count(collection: str) -> int:
    try:
        return int(get_vectorstore(collection)._collection.count())
    except Exception:
        return 0


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_vectorstore(collection: str = COLLECTION_DOCS) -> Chroma:
    return Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def get_retriever(*, k: int | None = None, collection: str = COLLECTION_DOCS):
    """Retriever LangChain para uma coleção (uso opcional)."""
    k = k or RAG_TOP_K
    return get_vectorstore(collection).as_retriever(search_kwargs={"k": k})


def search_merged(
    pergunta: str,
    *,
    k: int | None = None,
) -> list[tuple[Document, float]]:
    """Busca na coleção principal + refinamento, ordenada por similaridade."""
    k = k or RAG_TOP_K
    hits: list[tuple[Document, float]] = []
    hits.extend(get_vectorstore(COLLECTION_DOCS).similarity_search_with_score(pergunta, k=k))
    if _collection_count(COLLECTION_REFINE) > 0:
        refine_k = max(2, k // 2)
        hits.extend(
            get_vectorstore(COLLECTION_REFINE).similarity_search_with_score(
                pergunta, k=refine_k
            )
        )
    hits.sort(key=lambda x: x[1])
    return hits[:k]


def add_documents(docs: list[Document], *, collection: str = COLLECTION_DOCS) -> int:
    if not docs:
        return 0
    vs = get_vectorstore(collection)
    vs.add_documents(docs)
    return len(docs)


def index_text(
    text: str,
    *,
    source: str,
    extra_meta: dict | None = None,
    collection: str = COLLECTION_DOCS,
) -> int:
    from backend.ingest import text_to_documents

    docs = text_to_documents(text, source=source, extra_meta=extra_meta)
    return add_documents(docs, collection=collection)


def index_file(path: Path, *, collection: str = COLLECTION_DOCS) -> int:
    return add_documents(documents_from_path(path), collection=collection)


def index_all_knowledge(*, include_exemplos: bool = True) -> dict:
    counts = {"ficheiros": 0, "chunks": 0, "exemplos_chunks": 0}
    for path in iter_knowledge_files():
        n = index_file(path)
        counts["ficheiros"] += 1
        counts["chunks"] += n
    if include_exemplos:
        ex = load_exemplo_json()
        counts["exemplos_chunks"] = add_documents(ex)
        counts["chunks"] += counts["exemplos_chunks"]
    return counts


def similarity_search(pergunta: str, *, k: int | None = None, collection: str = COLLECTION_DOCS):
    """Atalho: embedding da pergunta + busca por similaridade."""
    k = k or RAG_TOP_K
    return get_vectorstore(collection).similarity_search_with_score(pergunta, k=k)


def collection_stats() -> dict:
    stats = {}
    for name in (COLLECTION_DOCS, COLLECTION_REFINE):
        try:
            stats[name] = get_vectorstore(name)._collection.count()
        except Exception:
            stats[name] = 0
    return stats
