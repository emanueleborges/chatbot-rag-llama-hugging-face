"""
Pipeline alinhado ao fluxo: PDF → PyPDFLoader → chunking semântico → texto RAG.

Os chunks são gravados em `knowledge/ev_knowledge.txt` (separador \\n\\n), compatível
com `rag_service` e ingestão Chroma local (embeddings Sentence-Transformers).

Embeddings OpenAI/Cohere ou Pinecone/Qdrant podem ser acrescentados à parte; este
projeto mantém Chroma + modelo multilingue configurável em `vector_store.py`.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from config import EV_KNOWLEDGE_FILE

_ROOT = Path(__file__).resolve().parent
_KNOWLEDGE = _ROOT / "knowledge"
_UPLOADS = _KNOWLEDGE / "pdf_uploads"


def _split_pdf_langchain(pdf_path: Path) -> list[str]:
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as e:
        raise RuntimeError(
            "Dependências LangChain em falta. Execute:\n"
            "  pip install langchain-community langchain-text-splitters pypdf"
        ) from e

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    splits = splitter.split_documents(docs)
    out: list[str] = []
    for d in splits:
        t = (d.page_content or "").strip()
        if t:
            out.append(t)
    return out


def ingest_pdf_to_ev_knowledge(
    pdf_path: Path,
    *,
    append: bool = True,
    ev_filename: str = EV_KNOWLEDGE_FILE,
) -> dict[str, Any]:
    """
    Extrai texto com LangChain, faz chunking e grava em knowledge/<ev_filename>.

    Se append=True, acrescenta uma secção nova; senão substitui só se o ficheiro
    estiver vazio (use `python rag_cli.py extract` ou extract_pdf.py para regeneração completa).
    """
    pdf_path = pdf_path.resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(str(pdf_path))

    chunks = _split_pdf_langchain(pdf_path)
    if not chunks:
        return {
            "ok": False,
            "error": "Nenhum texto extraído do PDF (ficheiro vazio ou só imagem?).",
            "chunks": 0,
        }

    section_header = (
        f"--- Upload LangChain · {pdf_path.name} · {uuid.uuid4().hex[:8]} ---\n"
        f"Chunks semânticos (PyPDFLoader + RecursiveCharacterTextSplitter).\n"
    )
    new_body = section_header + "\n\n".join(chunks)

    _KNOWLEDGE.mkdir(parents=True, exist_ok=True)
    target = _KNOWLEDGE / ev_filename

    if append and target.exists():
        prev = target.read_text(encoding="utf-8").strip()
        merged = f"{prev}\n\n{new_body}" if prev else new_body
    else:
        merged = new_body

    target.write_text(merged.strip() + "\n", encoding="utf-8")

    return {
        "ok": True,
        "ev_file": str(target.relative_to(_ROOT)),
        "sections_appended": 1,
        "chunks_written": len(chunks),
        "chars": len(new_body),
    }


def save_upload_to_pdf_uploads(file_bytes: bytes, original_name: str) -> Path:
    """Guarda upload bruto em knowledge/pdf_uploads/ com nome único."""
    _UPLOADS.mkdir(parents=True, exist_ok=True)
    stem = Path(original_name or "upload.pdf").name
    if not stem.lower().endswith(".pdf"):
        stem = stem + ".pdf"
    dest = _UPLOADS / f"{uuid.uuid4().hex}_{stem}"
    dest.write_bytes(file_bytes)
    return dest
