"""RAG por sobreposição de termos sobre ficheiros em knowledge/.

Não há modelo de embeddings treinável neste módulo: a «base treinada» são os .txt
em knowledge/. Use train_rag.py ou POST /api/rag/reload para validar e limpar cache."""
from __future__ import annotations

import re
from pathlib import Path

_KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
_chunks_cache: dict[str, list[str]] = {}


def load_chunks(filename: str = "dell_knowledge.txt") -> list[str]:
    path = _KNOWLEDGE_DIR / filename
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def get_chunks(filename: str = "dell_knowledge.txt") -> list[str]:
    """Chunks em memória (cache). Após editar knowledge/*.txt, chame clear_chunks_cache()."""
    if filename not in _chunks_cache:
        _chunks_cache[filename] = load_chunks(filename)
    return _chunks_cache[filename]


def clear_chunks_cache() -> None:
    """Limpa cache para o próximo pedido reler os ficheiros do disco."""
    _chunks_cache.clear()


def corpus_stats() -> dict[str, dict[str, int]]:
    """Estatísticas lidas diretamente do disco (ignora cache)."""
    out: dict[str, dict[str, int]] = {}
    for path in sorted(_KNOWLEDGE_DIR.glob("*.txt")):
        chunks = load_chunks(path.name)
        out[path.name] = {
            "chunks": len(chunks),
            "chars": sum(len(c) for c in chunks),
        }
    return out


def retrieve(query: str, chunks: list[str], top_k: int = 4) -> list[str]:
    if not chunks:
        return []
    q_words = set(re.findall(r"[\wàáâãéêíóôõúç]+", query.lower()))
    if not q_words:
        return chunks[:top_k]
    scored: list[tuple[float, str]] = []
    for ch in chunks:
        cw = set(re.findall(r"[\wàáâãéêíóôõúç]+", ch.lower()))
        overlap = len(q_words & cw)
        scored.append((float(overlap), ch))
    scored.sort(key=lambda x: -x[0])
    best = [c for s, c in scored[:top_k] if s > 0]
    if best:
        return best
    return [c for _, c in scored[:top_k]]


def retrieve_hybrid(query: str, source_file: str, chunks: list[str], top_k: int = 4) -> list[str]:
    """Chroma (por `source`) quando houver dados; senão recuperação por termos."""
    try:
        from vector_store import chroma_retrieve

        vec = chroma_retrieve(query, source_file, top_k)
        if vec:
            return vec
    except ImportError:
        pass
    except Exception:
        pass
    return retrieve(query, chunks, top_k)
