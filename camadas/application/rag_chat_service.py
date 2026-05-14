from __future__ import annotations

import asyncio
from typing import AsyncIterator, Sequence

from camadas.domain.chat import ChatLanguageModel, RAGAnswer, SourceSnippet
from camadas.domain.retriever import DocumentRetriever, KnowledgeStore, RetrievedChunk


def _chunk_text(
    text: str,
    *,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            newline = text.rfind("\n", start, end)
            if newline > start:
                end = newline + 1
            else:
                space = text.rfind(" ", start, end)
                if space > start:
                    end = space + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start = end - overlap if end < text_len else end
    return chunks


class RAGChatService:
    """
    Caso de uso: orquestra recuperação + prompt de RAG e delega geração ao LLM.
    """

    def __init__(
        self,
        *,
        retriever: DocumentRetriever,
        knowledge_store: KnowledgeStore,
        language_model: ChatLanguageModel,
        system_prompt: str | None = None,
        default_context_chunks: int = 5,
    ) -> None:
        self._retriever = retriever
        self._knowledge = knowledge_store
        self._llm = language_model
        self._k = default_context_chunks
        self._system_prompt = system_prompt or (
            "Você é um assistente útil que responde com base no contexto fornecido. "
            "Se a resposta não estiver no contexto, diga claramente que não encontrou a informação. "
            "Use linguagem clara e objetiva."
        )

    def _format_context(self, chunks: Sequence[RetrievedChunk]) -> str:
        parts = [c.content for c in chunks]
        return "\n\n---\n\n".join(parts)

    def _user_prompt(self, question: str, context: str) -> str:
        return (
            f"Contexto:\n{context}\n\n"
            f"Pergunta: {question}\n\n"
            "Responda com base somente no contexto acima."
        )

    def _to_sources(self, chunks: Sequence[RetrievedChunk]) -> tuple[SourceSnippet, ...]:
        return tuple(
            SourceSnippet(
                text=c.content[:500],
                score=c.score,
                metadata=dict(c.metadata) if c.metadata else None,
            )
            for c in chunks
        )

    def ingest_plain_text(
        self,
        text: str,
        *,
        base_metadata: dict | None = None,
    ) -> list[str]:
        base_metadata = dict(base_metadata or {})
        fragments = _chunk_text(text)
        metadatas = [
            {**base_metadata, "chunk_index": i, "total_chunks": len(fragments)}
            for i in range(len(fragments))
        ]
        return self._knowledge.add_texts(fragments, metadatas)

    def retrieve(self, query: str, *, k: int | None = None) -> list[RetrievedChunk]:
        return self._retriever.get_relevant_documents(query, k=k or self._k)

    async def answer(self, question: str) -> RAGAnswer:
        chunks = await asyncio.to_thread(self.retrieve, question)
        context = self._format_context(chunks)
        user_prompt = self._user_prompt(question, context)
        text = await self._llm.acomplete(
            system=self._system_prompt,
            user_prompt=user_prompt,
        )
        return RAGAnswer(text=text, sources=self._to_sources(chunks))

    async def answer_stream(self, question: str) -> AsyncIterator[str]:
        chunks = await asyncio.to_thread(self.retrieve, question)
        context = self._format_context(chunks)
        user_prompt = self._user_prompt(question, context)
        async for token in self._llm.astream_completion(
            system=self._system_prompt,
            user_prompt=user_prompt,
        ):
            yield token
