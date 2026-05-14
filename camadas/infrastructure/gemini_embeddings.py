from __future__ import annotations

import os
from typing import Sequence

import google.generativeai as genai
from langchain_core.embeddings import Embeddings


class GeminiGoogleEmbeddings(Embeddings):
    """
    Embeddings Google Gemini compatíveis com LangChain / Chroma.
    Usa `task_type` distinto para documentos vs consulta (melhor recuperação).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Defina GEMINI_API_KEY ou GOOGLE_API_KEY para usar embeddings Gemini.",
            )
        genai.configure(api_key=key)
        self._model = model or os.getenv(
            "GEMINI_EMBEDDING_MODEL",
            "models/text-embedding-004",
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for t in texts:
            res = genai.embed_content(
                model=self._model,
                content=t,
                task_type="retrieval_document",
            )
            vectors.append(list(res["embedding"]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        res = genai.embed_content(
            model=self._model,
            content=text,
            task_type="retrieval_query",
        )
        return list(res["embedding"])
