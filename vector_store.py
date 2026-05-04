"""ChromaDB persistente: embeddings multilingues e consulta por ficheiro-fonte."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
_CHROMA_DIR = Path(os.environ.get("CHROMA_PATH", str(_ROOT / "chroma_db")))
_COLLECTION_NAME = "rag_knowledge"
_MODEL = os.environ.get(
    "CHROMA_EMBED_MODEL",
    "paraphrase-multilingual-MiniLM-L12-v2",
)

_client: Any = None
_collection: Any = None


def chroma_persist_path() -> Path:
    return _CHROMA_DIR


def _embedding_function():
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return SentenceTransformerEmbeddingFunction(model_name=_MODEL)


def reset_vector_store() -> None:
    global _client, _collection
    _client = None
    _collection = None


def get_collection():
    """Coleção única com metadado `source` = nome do ficheiro em knowledge/."""
    global _client, _collection
    if _collection is not None:
        return _collection
    import chromadb

    _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(_CHROMA_DIR.resolve()))
    _collection = _client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=_embedding_function(),
    )
    return _collection


def chroma_count() -> int | None:
    try:
        return int(get_collection().count())
    except Exception:
        return None


def chroma_retrieve(query: str, source: str, top_k: int = 4) -> list[str] | None:
    """
    Devolve trechos mais similares à pergunta para aquele `source`, ou None
    se Chroma vazio / erro / sem resultados.
    """
    if not (query or "").strip():
        return None
    try:
        coll = get_collection()
        if coll.count() == 0:
            return None
        res = coll.query(
            query_texts=[query.strip()],
            n_results=min(top_k, max(1, coll.count())),
            where={"source": source},
        )
        docs = (res or {}).get("documents") or []
        if not docs or not docs[0]:
            return None
        out = [d for d in docs[0] if isinstance(d, str) and d.strip()]
        return out or None
    except Exception:
        return None


def ingest_all_knowledge_txt() -> dict[str, Any]:
    """
    Reconstrói a coleção a partir de todos os `knowledge/*.txt` (mesmos chunks que rag_service).
    Apaga a coleção anterior para evitar IDs órfãos.
    """
    from rag_service import load_chunks

    import chromadb

    knowledge_dir = _ROOT / "knowledge"

    _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    ef = _embedding_function()
    client = chromadb.PersistentClient(path=str(_CHROMA_DIR.resolve()))
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass

    coll = client.create_collection(
        name=_COLLECTION_NAME,
        embedding_function=ef,
    )
    global _client, _collection
    _client = client
    _collection = coll

    stats: dict[str, int] = {}
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, str]] = []

    for path in sorted(knowledge_dir.glob("*.txt")):
        name = path.name
        chunks = load_chunks(name)
        stats[name] = len(chunks)
        for i, doc in enumerate(chunks):
            ids.append(f"{name}:{i}")
            documents.append(doc)
            metadatas.append({"source": name})

    if ids:
        batch = 128
        for start in range(0, len(ids), batch):
            end = start + batch
            coll.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

    return {
        "chroma_path": str(_CHROMA_DIR.resolve()),
        "collection": _COLLECTION_NAME,
        "embed_model": _MODEL,
        "chunks_by_file": stats,
        "total_chunks": len(ids),
    }
