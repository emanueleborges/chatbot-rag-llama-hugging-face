from __future__ import annotations

import threading

from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def messages_to_chat_dicts(messages: list[BaseMessage]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            out.append({"role": "assistant", "content": m.content})
    return out


def format_history_for_condense(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"Usuário: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistente: {m.content}")
    return "\n".join(lines)


class SessionWindowMemoryStore:
    """
    Uma `ConversationBufferWindowMemory` por sessão (últimas k interações).
    """

    def __init__(self, *, window_interactions: int = 5) -> None:
        self._k = window_interactions
        self._lock = threading.Lock()
        self._sessions: dict[str, ConversationBufferWindowMemory] = {}

    def get_memory(self, session_id: str) -> ConversationBufferWindowMemory:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = ConversationBufferWindowMemory(
                    k=self._k,
                    return_messages=True,
                    memory_key="history",
                )
            return self._sessions[session_id]

    def history_dicts(self, session_id: str) -> list[dict[str, str]]:
        mem = self.get_memory(session_id)
        msgs = mem.load_memory_variables({}).get("history") or []
        return messages_to_chat_dicts(list(msgs))
