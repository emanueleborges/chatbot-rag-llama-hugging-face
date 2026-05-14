from __future__ import annotations

import os

from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings

from camadas.domain.embeddings import EmbeddingsPort


def build_embeddings_from_env() -> EmbeddingsPort:
    """
    Preferência por OpenAI quando `OPENAI_API_KEY` está definida;
    caso contrário, embeddings locais via HuggingFace (Sentence-Transformers).
    """
    if os.getenv("OPENAI_API_KEY"):
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)
    model_name = os.getenv(
        "HF_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    return HuggingFaceEmbeddings(model_name=model_name)
