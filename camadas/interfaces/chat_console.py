from __future__ import annotations

from camadas.application.conversational_rag_service import ConversationalRAGService


async def enviar_pergunta_e_exibir_resposta(
    servico: ConversationalRAGService,
    *,
    session_id: str,
    pergunta: str,
    usar_stream: bool = False,
) -> None:
    """
    Envia uma pergunta ao chatbot e imprime a resposta junto com o histórico da sessão.
    Útil para testes locais ou scripts de console.
    """
    if usar_stream:
        print("=== Histórico (antes) ===")
        for msg in servico.get_session_history(session_id):
            print(f"{msg['role']}: {msg['content']}")
        print("=== Resposta (stream) ===")
        async for trecho in servico.converse_stream(session_id, pergunta):
            print(trecho, end="", flush=True)
        print()
        print("=== Histórico (depois) ===")
        for msg in servico.get_session_history(session_id):
            print(f"{msg['role']}: {msg['content']}")
        return

    resultado = await servico.converse(session_id, pergunta)
    print("=== Pergunta autossuficiente (reformulada) ===")
    print(resultado.standalone_question)
    print("=== Histórico da conversa ===")
    for msg in resultado.history:
        print(f"{msg['role']}: {msg['content']}")
    print("=== Resposta ===")
    print(resultado.answer)
    if resultado.input_blocked or resultado.output_blocked:
        print("=== Aviso de guardrail ===")
        print(resultado.block_reason or "Conteúdo filtrado.")
