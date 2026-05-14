from __future__ import annotations

import os

from dotenv import load_dotenv

from camadas.application.conversational_rag_service import ConversationalRAGService
from camadas.infrastructure.basic_guardrails import BasicGuardrails
from camadas.infrastructure.cross_encoder_reranker import (
    CrossEncoderDocumentReranker,
    IdentityDocumentReranker,
)
from camadas.infrastructure.langchain_retriever import LangChainVectorRetriever
from camadas.infrastructure.ollama_llm import OllamaLangChainModel
from camadas.infrastructure.openai_embeddings import build_embeddings_from_env
from camadas.infrastructure.query_rewriter import HistoryAwareQueryRewriter
from camadas.infrastructure.session_memory import SessionWindowMemoryStore
from camadas.infrastructure.vector_store import ChromaKnowledgeStore, build_chroma_vector_store
from camadas.interfaces.api import AppComposition, create_app
from image_gen import ImageGenerator

load_dotenv()


def _has_gemini_credentials() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def _optional_legacy_cache_clear():
    if os.getenv("ENABLE_LEGACY_REDIS_CACHE_CLEAR", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return None
    from database import VectorDatabase

    legacy = VectorDatabase()

    def _clear() -> None:
        legacy.clear_cache()

    return _clear


def build_application():
    if _has_gemini_credentials():
        from camadas.infrastructure.gemini_embeddings import GeminiGoogleEmbeddings
        from camadas.infrastructure.gemini_llm import GeminiChatLanguageModel

        embeddings = GeminiGoogleEmbeddings()
        llm = GeminiChatLanguageModel()
        default_collection = "rag_gemini"
    else:
        embeddings = build_embeddings_from_env()
        llm = OllamaLangChainModel()
        default_collection = "rag_langchain"

    collection = os.getenv("CHROMA_COLLECTION") or default_collection
    chroma = build_chroma_vector_store(
        embeddings,
        collection_name=collection,
    )
    store = ChromaKnowledgeStore(chroma)
    fetch_k = int(os.getenv("RAG_FETCH_K", "20"))
    retriever = LangChainVectorRetriever(chroma, default_k=fetch_k)

    if os.getenv("DISABLE_RERANKER", "").lower() in ("1", "true", "yes"):
        reranker = IdentityDocumentReranker()
    else:
        reranker = CrossEncoderDocumentReranker()

    window = int(os.getenv("CONV_BUFFER_WINDOW", "5"))
    memory_store = SessionWindowMemoryStore(window_interactions=window)
    guards = BasicGuardrails()
    rewriter = HistoryAwareQueryRewriter(llm)

    rag = ConversationalRAGService(
        knowledge_store=store,
        retriever=retriever,
        reranker=reranker,
        language_model=llm,
        query_rewriter=rewriter,
        memory_store=memory_store,
        input_guard=guards,
        output_guard=guards,
    )
    composition = AppComposition(
        rag_service=rag,
        image_generator=ImageGenerator(),
        clear_cache=_optional_legacy_cache_clear(),
    )
    return create_app(composition)


app = build_application()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
