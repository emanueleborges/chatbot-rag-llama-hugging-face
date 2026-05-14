from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from camadas.domain.chat import SourceSnippet


@dataclass(frozen=True)
class ConversationRAGResult:
    """Resposta do RAG conversacional com metadados de sessão e reformulação."""

    answer: str
    sources: Tuple[SourceSnippet, ...]
    standalone_question: str
    history: Tuple[dict[str, str], ...]
    input_blocked: bool = False
    output_blocked: bool = False
    block_reason: str | None = None
