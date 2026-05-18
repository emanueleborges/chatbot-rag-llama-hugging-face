"""Memória conversacional por sessão (janela deslizante, só langchain-core)."""
from __future__ import annotations

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from backend.config import MEMORY_WINDOW


class _SessionHistory(BaseChatMessageHistory):
    def __init__(self) -> None:
        self._messages: list[BaseMessage] = []

    @property
    def messages(self) -> list[BaseMessage]:
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)

    def clear(self) -> None:
        self._messages = []


_store: dict[str, _SessionHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = _SessionHistory()
    return _store[session_id]


def trim_history(session_id: str) -> None:
    hist = get_session_history(session_id)
    msgs = hist.messages
    if len(msgs) <= MEMORY_WINDOW * 2:
        return
    hist.clear()
    for m in msgs[-(MEMORY_WINDOW * 2) :]:
        hist.add_message(m)


def clear_session(session_id: str) -> None:
    _store.pop(session_id, None)
