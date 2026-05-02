"""RAG simples por sobreposição de termos sobre documentos Dell."""
from __future__ import annotations

import re
from pathlib import Path

_KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"


def load_chunks(filename: str = "dell_knowledge.txt") -> list[str]:
    path = _KNOWLEDGE_DIR / filename
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [p.strip() for p in text.split("\n\n") if p.strip()]


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
