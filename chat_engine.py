"""Núcleo do chat (RAG, LLM, imagem) — usado pela API FastAPI."""
from __future__ import annotations

import asyncio

from image_service import generate_image_b64
from llm_service import chat_completion
from rag_service import load_chunks, retrieve
from routing import extract_image_prompt, route_message

ROUTE_LABELS = {
    "rag": "📄 RAG — Documentos Dell",
    "rag_manaus": "📄 RAG — Manaus (local)",
    "llm": "🤖 LLM — Ollama",
    "image": "🎨 Imagem — HF ou Pollinations",
}

_SYSTEM_MANAUS = (
    "Você responde sobre locais e geografia em torno de Manaus (AM) usando só o contexto. "
    "Proibido inventar distâncias em km ou afirmar que Itacoatiara (ou outro município) fica "
    "‘a 25 km’ de Manaus se o contexto disser o contrário. Para km exatos, mande o utilizador "
    "confirmar em mapa ou IBGE. Use APENAS nomes e factos do contexto. Não cite shoppings ou "
    "praças de outras capitais. Para ‘melhor shopping’ ou ‘melhor praça’, diga que é subjetivo."
)

_SYSTEM_LLM = (
    "Você é o **RAG Chat**, assistente amigável. Responda em português do Brasil, "
    "claro e objetivo. Você pode conversar sobre qualquer assunto quando o usuário não "
    "estiver perguntando especificamente documentados.\n\n"
    "**Locais e cidades:** se o usuário citar uma cidade ou região (ex.: Manaus), só mencione "
    "lugares, shoppings, endereços ou estabelecimentos que existam de fato nessa cidade — não "
    "misture com outros estados ou invente nomes. Para perguntas do tipo \"o melhor shopping\", "
    "\"onde ir\", rankings ou dados que mudam com o tempo, não invente certezas: diga que não "
    "há uma única resposta objetiva sem dados atualizados e sugira conferir no Google Maps, "
    "sites oficiais ou avaliações recentes na própria cidade. Se não tiver informação confiável "
    "sobre aquele lugar, admita e evite listar estabelecimentos que possam ser de outra cidade."
)

_chunks_cache: dict[str, list[str]] = {}


def get_chunks(filename: str = "dell_knowledge.txt") -> list[str]:
    if filename not in _chunks_cache:
        _chunks_cache[filename] = load_chunks(filename)
    return _chunks_cache[filename]


async def process_chat(text: str, history: list[dict]) -> dict:
    """
    history: mensagens anteriores [{"role":"user"|"assistant","content":"..."}, ...]
    Retorno igual ao JSON da API /api/chat.
    """
    text = (text or "").strip()
    if not text:
        return {
            "content": "Digite uma mensagem.",
            "route": "llm",
            "route_label": ROUTE_LABELS["llm"],
            "image": None,
            "error": True,
        }

    route = route_message(text)
    label = ROUTE_LABELS[route]
    prior = history[-24:] if history else []

    try:
        if route == "image":
            prompt = extract_image_prompt(text)
            if not prompt:
                prompt = text
            mime, b64 = await generate_image_b64(prompt)
            return {
                "content": f"Segue a imagem gerada para o pedido: **{prompt}**",
                "route": route,
                "route_label": label,
                "image": {"mime": mime, "base64": b64},
                "error": False,
            }

        if route == "rag_manaus":
            chunks = get_chunks("manaus_local.txt")
            if not chunks:
                reply = (
                    "Base **Manaus** não encontrada no servidor (`knowledge/manaus_local.txt`). "
                    "Peça ao administrador para verificar o ficheiro."
                )
                return {
                    "content": reply,
                    "route": route,
                    "route_label": label,
                    "image": None,
                    "error": True,
                }
            context = "\n\n".join(chunks)
            system = (
                f"{_SYSTEM_MANAUS}\n\n--- CONTEXTO (Manaus, AM) ---\n{context}\n--- FIM DO CONTEXTO ---"
            )
            messages = [{"role": "system", "content": system}] + prior + [{"role": "user", "content": text}]
        elif route == "rag":
            chunks = get_chunks("dell_knowledge.txt")
            picked = retrieve(text, chunks, top_k=4)
            context = "\n\n".join(picked) if picked else "(sem trechos recuperados)"
            system = (
                "Você é um assistente especializado em produtos e suporte Dell. "
                "Use principalmente o contexto abaixo (documentos internos). "
                "Se algo não constar no contexto, diga honestamente. Responda em português do Brasil.\n\n"
                f"--- CONTEXTO ---\n{context}\n--- FIM DO CONTEXTO ---"
            )
            messages = [{"role": "system", "content": system}] + prior + [{"role": "user", "content": text}]
        else:
            messages = [{"role": "system", "content": _SYSTEM_LLM}] + prior + [{"role": "user", "content": text}]

        reply = await asyncio.to_thread(chat_completion, messages)
        return {
            "content": reply,
            "route": route,
            "route_label": label,
            "image": None,
            "error": False,
        }
    except Exception as exc:
        return {
            "content": f"**Erro:** {exc}",
            "route": route,
            "route_label": label,
            "image": None,
            "error": True,
        }


def format_assistant_message(result: dict) -> str:
    """Texto + imagem em Markdown (`![](data:...)`)."""
    text = result.get("content") or ""
    img = result.get("image")
    if img and img.get("base64") and img.get("mime"):
        text += f"\n\n![](data:{img['mime']};base64,{img['base64']})"
    return text
