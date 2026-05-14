from __future__ import annotations

import sys


def _walk_exceptions(exc: BaseException | None) -> list[BaseException]:
    if exc is None:
        return []
    out: list[BaseException] = [exc]
    if sys.version_info >= (3, 11) and isinstance(exc, BaseExceptionGroup):
        for child in exc.exceptions:
            out.extend(_walk_exceptions(child))
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, BaseException):
        out.extend(_walk_exceptions(cause))
    return out


def is_llm_transport_error(exc: BaseException) -> bool:
    """Deteta falhas típicas de rede ao contactar Ollama / LLM local."""
    try:
        from aiohttp import ClientConnectorError, ClientOSError, ServerDisconnectedError
    except ImportError:  # pragma: no cover
        ClientConnectorError = ClientOSError = ServerDisconnectedError = None  # type: ignore

    for e in _walk_exceptions(exc):
        if isinstance(e, (ConnectionRefusedError, ConnectionResetError, TimeoutError)):
            return True
        if isinstance(e, OSError) and getattr(e, "winerror", None) in (1225, 10061):
            return True
        if ClientConnectorError is not None and isinstance(
            e,
            (ClientConnectorError, ClientOSError, ServerDisconnectedError),
        ):
            return True
    return False


def friendly_llm_unreachable_message(*, backend: str, endpoint: str) -> str:
    return (
        f"Não foi possível ligar ao {backend} em `{endpoint}`. "
        "Inicie o serviço (por exemplo `ollama serve` no Windows) ou use Docker na porta 11434. "
        "Alternativa: defina `GEMINI_API_KEY` ou `GOOGLE_API_KEY` no `.env` para usar o Google Gemini."
    )
