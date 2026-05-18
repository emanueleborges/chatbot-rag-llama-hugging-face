"""Chat RAG — delega ao pipeline de 3 fases (sem langchain.chains)."""
from __future__ import annotations

from backend.rag_pipeline import executar_rag


def chat(
    message: str,
    *,
    session_id: str = "default",
    modo: str = "pipeline",
) -> dict:
    """Recuperação + geração com memória conversacional."""
    del modo  # mantido por compatibilidade com a API
    out = executar_rag(message, session_id=session_id)
    return {
        "answer": out["resposta"],
        "sources": out.get("sources", []),
        "recuperacao": out.get("recuperacao"),
        "geracao": out.get("geracao"),
    }
