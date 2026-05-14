from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class EmbeddingsPort(Protocol):
    """Porta de embeddings alinhada ao contrato típico do LangChain."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Vetores para uma lista de textos (ingestão)."""

    def embed_query(self, text: str) -> list[float]:
        """Vetor para uma consulta (recuperação)."""
