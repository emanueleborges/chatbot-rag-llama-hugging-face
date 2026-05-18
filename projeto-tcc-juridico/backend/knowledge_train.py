"""Treino / indexação da base de conhecimento (PDFs em data/knowledge/)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from backend.config import CHROMA_DIR, COLLECTION_DOCS, DATA_DIR, KNOWLEDGE_DIR
from backend.ingest import iter_knowledge_files
from backend.rag_engine import collection_stats, get_vectorstore
from backend.rag_pipeline import indexar_arquivo

MANIFEST_PATH = DATA_DIR / "treino_manifest.json"


def _listar_pdfs_knowledge() -> list[Path]:
    if not KNOWLEDGE_DIR.is_dir():
        return []
    paths: list[Path] = []
    for ext in ("*.pdf", "*.txt", "*.md", "*.docx"):
        paths.extend(KNOWLEDGE_DIR.glob(ext))
    return sorted(set(paths))


def reset_colecao_docs() -> None:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_DOCS)
    except Exception:
        pass
    get_vectorstore(COLLECTION_DOCS)


def treinar_base_conhecimento(
    *,
    reset: bool = True,
    apenas_knowledge: bool = True,
) -> dict:
    """
    Indexa todos os PDFs (e TXT/DOCX) de data/knowledge/ no ChromaDB.
    Com reset=True, apaga a coleção anterior antes de reindexar.
    """
    if reset:
        reset_colecao_docs()

    paths = _listar_pdfs_knowledge() if apenas_knowledge else iter_knowledge_files()
    if not paths:
        return {
            "status": "erro",
            "mensagem": f"Nenhum ficheiro em {KNOWLEDGE_DIR}. Coloque os PDFs das petições lá.",
            "ficheiros": [],
        }

    detalhes: list[dict] = []
    total_chunks = 0
    erros: list[str] = []

    for path in paths:
        try:
            r = indexar_arquivo(path)
            total_chunks += r["chunks"]
            detalhes.append(
                {
                    "arquivo": path.name,
                    "chunks": r["chunks"],
                    "caracteres": r["caracteres"],
                }
            )
        except Exception as e:
            erros.append(f"{path.name}: {e}")

    resultado = {
        "status": "ok" if detalhes else "erro",
        "treinado_em": datetime.now(timezone.utc).isoformat(),
        "pasta": str(KNOWLEDGE_DIR),
        "reset_aplicado": reset,
        "ficheiros_indexados": len(detalhes),
        "total_chunks": total_chunks,
        "detalhes": detalhes,
        "erros": erros,
        "chroma": collection_stats(),
    }

    MANIFEST_PATH.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return resultado


def status_treino() -> dict:
    manifest = {}
    if MANIFEST_PATH.is_file():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    pdfs = [p.name for p in _listar_pdfs_knowledge()]
    return {
        "pdfs_em_knowledge": pdfs,
        "total_pdfs": len(pdfs),
        "ultimo_treino": manifest,
        "chroma": collection_stats(),
    }
