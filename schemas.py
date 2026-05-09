"""Contratos HTTP da API — camada de interface (FastAPI / JSON)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = ""
    history: list[ChatMessage] = Field(default_factory=list)
    beginner_mode: bool | None = Field(
        default=None,
        description="None=detetar automaticamente; True=modo Primeira viagem; False=desligado",
    )
    strict_grounding: bool | None = Field(
        default=None,
        description="None=usar env RAG_STRICT_GROUNDING; True=só resposta com trechos recuperados (sem fallback web/LLM)",
    )
