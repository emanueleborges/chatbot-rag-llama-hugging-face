from __future__ import annotations

from camadas.domain.chat import ChatLanguageModel
from camadas.infrastructure.llm_transport_errors import is_llm_transport_error


class HistoryAwareQueryRewriter:
    """Reformula a pergunta atual em uma consulta autossuficiente dado o histórico."""

    def __init__(self, llm: ChatLanguageModel) -> None:
        self._llm = llm
        self._system = (
            "Você reescreve perguntas de seguimento para uma pergunta única e autossuficiente, "
            "em português, preservando a intenção do usuário. "
            "Se já for autossuficiente, repita-a com clareza. "
            "Responda somente com a pergunta reformulada, sem explicações."
        )

    async def rewrite(self, history_text: str, latest_question: str) -> str:
        if not history_text.strip():
            return latest_question.strip()
        user = (
            f"Histórico da conversa:\n{history_text}\n\n"
            f"Última mensagem do usuário:\n{latest_question}\n\n"
            "Reescreva a última mensagem como pergunta autossuficiente:"
        )
        try:
            out = (await self._llm.acomplete(system=self._system, user_prompt=user)).strip()
        except BaseException as exc:
            if is_llm_transport_error(exc):
                return latest_question.strip()
            raise
        return out or latest_question.strip()
