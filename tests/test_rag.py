"""Testes do RAG por termos (Chroma desativado via mock — CI rápido)."""
from rag_service import load_chunks, retrieve, retrieve_hybrid


def test_load_dell_chunks():
    chunks = load_chunks("dell_knowledge.txt")
    assert len(chunks) >= 1


def test_retrieve_overlap():
    chunks = load_chunks("dell_knowledge.txt")
    out = retrieve("garantia service tag Brasil", chunks, top_k=3)
    assert len(out) >= 1
    joined = " ".join(out).lower()
    assert "garantia" in joined or "service" in joined or "dell" in joined


def test_retrieve_hybrid_fallback(monkeypatch):
    """Força fallback por termos sem carregar Chroma / Sentence-Transformers."""
    monkeypatch.setattr("vector_store.chroma_retrieve", lambda *a, **k: None)
    chunks = load_chunks("lenovo_knowledge.txt")
    out = retrieve_hybrid("ThinkPad BIOS drivers", "lenovo_knowledge.txt", chunks, top_k=2)
    assert len(out) >= 1
