from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator, Sequence

from camadas.application.rag_chat_service import _chunk_text
from camadas.domain.chat import ChatLanguageModel, SourceSnippet
from camadas.domain.conversation import ConversationRAGResult
from camadas.domain.guardrails import InputSafetyPolicy, OutputSafetyPolicy
from camadas.domain.rerank import DocumentReranker
from camadas.domain.retriever import DocumentRetriever, KnowledgeStore, RetrievedChunk
from camadas.infrastructure.langchain_retrieval_runnable import (
    build_retrieve_and_rerank_runnable,
)
from camadas.infrastructure.query_rewriter import HistoryAwareQueryRewriter
from camadas.infrastructure.session_memory import (
    SessionWindowMemoryStore,
    format_history_for_condense,
)


class ConversationalRAGService:
    """
    RAG com histórico em janela, reformulação de pergunta, re-ranking e guardrails.
    """

    def __init__(
        self,
        *,
        knowledge_store: KnowledgeStore,
        retriever: DocumentRetriever,
        reranker: DocumentReranker,
        language_model: ChatLanguageModel,
        query_rewriter: HistoryAwareQueryRewriter,
        memory_store: SessionWindowMemoryStore,
        input_guard: InputSafetyPolicy,
        output_guard: OutputSafetyPolicy,
        system_prompt: str | None = None,
        fetch_k: int | None = None,
        final_k: int | None = None,
    ) -> None:
        self._knowledge = knowledge_store
        self._llm = language_model
        self._rewriter = query_rewriter
        self._memory = memory_store
        self._input_guard = input_guard
        self._output_guard = output_guard
        self._fetch_k = int(os.getenv("RAG_FETCH_K", "20")) if fetch_k is None else fetch_k
        self._final_k = int(os.getenv("RAG_FINAL_K", "5")) if final_k is None else final_k
        self._system_prompt = system_prompt or (
            "Você é um assistente útil que responde com base no contexto fornecido. "
            "Se a resposta não estiver no contexto, diga claramente que não encontrou a informação. "
            "Use linguagem clara e objetiva."
        )
        self._retrieval_runnable = build_retrieve_and_rerank_runnable(
            retriever,
            reranker,
            fetch_k=self._fetch_k,
            final_k=self._final_k,
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

    def get_session_history(self, session_id: str) -> list[dict[str, str]]:
        return self._memory.history_dicts(session_id)

    def _format_context(self, chunks: Sequence[RetrievedChunk]) -> str:
        return "\n\n---\n\n".join(c.content for c in chunks)

    def _user_prompt(self, original_question: str, context: str) -> str:
        return (
            f"Contexto:\n{context}\n\n"
            f"Pergunta do usuário:\n{original_question}\n\n"
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

    async def converse(self, session_id: str, message: str) -> ConversationRAGResult:
        verdict_in = self._input_guard.evaluate_input(message)
        history_before = tuple(self._memory.history_dicts(session_id))
        if not verdict_in.allowed:
            return ConversationRAGResult(
                answer="Não posso processar esta mensagem.",
                sources=(),
                standalone_question=message.strip(),
                history=history_before,
                input_blocked=True,
                block_reason=verdict_in.reason,
            )

        working = verdict_in.safe_text or message.strip()
        mem = self._memory.get_memory(session_id)
        hist_msgs = mem.load_memory_variables({}).get("history") or []
        hist_text = format_history_for_condense(list(hist_msgs))
        standalone = await self._rewriter.rewrite(hist_text, working)

        reranked = await asyncio.to_thread(
            self._retrieval_runnable.invoke,
            {"standalone_query": standalone},
        )

        context = self._format_context(reranked)
        user_prompt = self._user_prompt(working, context)
        answer = await self._llm.acomplete(
            system=self._system_prompt,
            user_prompt=user_prompt,
        )

        verdict_out = self._output_guard.evaluate_output(answer)
        if verdict_out.allowed:
            final = verdict_out.safe_text or answer
            output_blocked = False
        else:
            final = verdict_out.safe_text or (
                "A resposta foi filtrada por políticas de segurança."
            )
            output_blocked = True

        mem.save_context({"input": working}, {"output": final})
        history_after = tuple(self._memory.history_dicts(session_id))

        return ConversationRAGResult(
            answer=final,
            sources=self._to_sources(reranked),
            standalone_question=standalone,
            history=history_after,
            input_blocked=False,
            output_blocked=output_blocked,
            block_reason=verdict_out.reason if output_blocked else None,
        )

    async def converse_stream(
        self,
        session_id: str,
        message: str,
        *,
        chunk_size: int = 48,
    ) -> AsyncIterator[str]:
        verdict_in = self._input_guard.evaluate_input(message)
        if not verdict_in.allowed:
            yield "Não posso processar esta mensagem."
            return

        working = verdict_in.safe_text or message.strip()
        mem = self._memory.get_memory(session_id)
        hist_msgs = mem.load_memory_variables({}).get("history") or []
        hist_text = format_history_for_condense(list(hist_msgs))
        standalone = await self._rewriter.rewrite(hist_text, working)

        reranked = await asyncio.to_thread(
            self._retrieval_runnable.invoke,
            {"standalone_query": standalone},
        )

        context = self._format_context(reranked)
        user_prompt = self._user_prompt(working, context)

        parts: list[str] = []
        async for token in self._llm.astream_completion(
            system=self._system_prompt,
            user_prompt=user_prompt,
        ):
            parts.append(token)

        full = "".join(parts)
        verdict_out = self._output_guard.evaluate_output(full)
        final = (
            (verdict_out.safe_text or full)
            if verdict_out.allowed
            else (
                verdict_out.safe_text
                or "A resposta foi filtrada por políticas de segurança."
            )
        )

        for i in range(0, len(final), max(1, chunk_size)):
            yield final[i : i + chunk_size]

        mem.save_context({"input": working}, {"output": final})
