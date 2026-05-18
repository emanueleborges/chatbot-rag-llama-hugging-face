"""
Pipeline RAG em 3 fases (petições jurídicas em PDF/TXT/DOCX):

  1. INDEXAÇÃO   PDF → Documento → Chunks → Embeddings → ChromaDB
  2. RECUPERAÇÃO Pergunta → Embedding → Similaridade → Chunks relevantes
  3. GERAÇÃO     Chunks + Pergunta (+ histórico) → Ollama → Resposta fundamentada
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from backend.config import RAG_TOP_K
from backend.ingest import (
    documents_from_path,
    extract_path,
    text_to_documents,
)
from backend.memory import get_session_history, trim_history
from backend.rag_engine import (
    COLLECTION_DOCS,
    add_documents,
    get_embeddings,
    get_vectorstore,
)

# --- Fase 1: Indexação -------------------------------------------------------

def indexar_arquivo(
    path: Path | str,
    *,
    collection: str = COLLECTION_DOCS,
) -> dict[str, Any]:
    """
    PDF/TXT/DOCX → texto → chunks (Document) → embeddings → banco vetorial.
    """
    path = Path(path)
    texto = extract_path(path)
    chunks_docs = documents_from_path(path)
    n = add_documents(chunks_docs, collection=collection)
    return {
        "fase": "indexacao",
        "arquivo": path.name,
        "caracteres": len(texto),
        "chunks": n,
        "coleção": collection,
        "etapas": [
            "1. Extração de texto do ficheiro",
            "2. Divisão em chunks (RecursiveCharacterTextSplitter)",
            "3. Geração de embeddings (Sentence Transformers)",
            "4. Persistência no ChromaDB",
        ],
    }


def indexar_texto(
    texto: str,
    *,
    source: str = "texto_manual",
    extra_meta: dict | None = None,
    collection: str = COLLECTION_DOCS,
) -> dict[str, Any]:
    chunks_docs = text_to_documents(texto, source=source, extra_meta=extra_meta)
    n = add_documents(chunks_docs, collection=collection)
    return {
        "fase": "indexacao",
        "source": source,
        "caracteres": len(texto),
        "chunks": n,
        "coleção": collection,
    }


# --- Fase 2: Recuperação -----------------------------------------------------

def recuperar_chunks(
    pergunta: str,
    *,
    k: int | None = None,
    collection: str = COLLECTION_DOCS,
) -> dict[str, Any]:
    """
    Pergunta → embedding da pergunta → busca por similaridade (cosseno) → top-k chunks.
    """
    k = k or RAG_TOP_K
    vs = get_vectorstore(collection)
    if vs._collection.count() == 0:
        return {
            "fase": "recuperacao",
            "pergunta": pergunta,
            "chunks": [],
            "aviso": "Base vazia. Indexe PDFs em data/knowledge/ ou use POST /rag/indexar.",
        }

    # LangChain/Chroma: embed da query + similaridade
    resultados = vs.similarity_search_with_score(pergunta, k=k)
    chunks = []
    for doc, distancia in resultados:
        # Chroma devolve distância (menor = mais similar); relevância ≈ 1 - dist
        relevancia = max(0.0, 1.0 - float(distancia))
        chunks.append(
            {
                "texto": doc.page_content,
                "metadata": doc.metadata,
                "distancia": round(float(distancia), 4),
                "relevancia": round(relevancia, 4),
            }
        )
    return {
        "fase": "recuperacao",
        "pergunta": pergunta,
        "k": k,
        "coleção": collection,
        "modelo_embedding": get_embeddings().model_name,
        "chunks": chunks,
        "etapas": [
            "1. Embedding da pergunta (mesmo modelo dos documentos)",
            "2. Busca por similaridade no ChromaDB",
            "3. Retorno dos top-k chunks mais próximos",
        ],
    }


# --- Fase 3: Geração ---------------------------------------------------------

_PROMPT_GERACAO = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Você é assistente jurídico para petições. Responda APENAS com base nos trechos abaixo.
Se o contexto não bastar, indique o que falta. Não invente jurisprudência nem artigos.
Responda em português do Brasil, de forma clara e estruturada.

TRECHOS RECUPERADOS:
{contexto}""",
        ),
        MessagesPlaceholder("historico"),
        ("human", "{pergunta}"),
    ]
)


def gerar_resposta(
    pergunta: str,
    chunks: list[dict[str, Any]] | list[Document],
    *,
    chat_history: list | None = None,
) -> dict[str, Any]:
    """
    Chunks relevantes + pergunta → LLM (Ollama) → resposta fundamentada.
    """
    if chunks and isinstance(chunks[0], Document):
        docs = chunks
    else:
        docs = [
            Document(
                page_content=c.get("texto", c.get("page_content", "")),
                metadata=c.get("metadata", {}),
            )
            for c in (chunks or [])
        ]

    if not docs:
        return {
            "fase": "geracao",
            "pergunta": pergunta,
            "resposta": (
                "Não há documentos indexados para fundamentar a resposta. "
                "Envie petições em PDF para data/knowledge/ e execute a indexação."
            ),
            "chunks_usados": 0,
        }

    contexto = "\n\n---\n\n".join(
        f"[Fonte: {d.metadata.get('source', '?')}]\n{d.page_content}"
        for d in docs
    )
    from backend.llm_local import get_chat_llm

    llm = get_chat_llm()
    historico = chat_history or []
    chain = _PROMPT_GERACAO | llm
    msg = chain.invoke(
        {
            "contexto": contexto,
            "historico": historico,
            "pergunta": pergunta,
        }
    )
    resposta = msg.content if hasattr(msg, "content") else str(msg)
    return {
        "fase": "geracao",
        "pergunta": pergunta,
        "resposta": resposta.strip(),
        "chunks_usados": len(docs),
        "fontes": [
            {
                "source": d.metadata.get("source", ""),
                "trecho": (d.page_content or "")[:200],
            }
            for d in docs
        ],
        "etapas": [
            "1. Montagem do prompt com chunks recuperados",
            "2. Chamada ao LLM local (Ollama)",
            "3. Resposta fundamentada nos trechos",
        ],
    }


# --- Pipeline completo (chat) ------------------------------------------------

def executar_rag(
    pergunta: str,
    *,
    session_id: str = "default",
    k: int | None = None,
    usar_memoria: bool = True,
) -> dict[str, Any]:
    """Executa as 3 fases em sequência e actualiza memória conversacional."""
    hist_msgs = []
    if usar_memoria:
        hist_msgs = get_session_history(session_id).messages

    rec = recuperar_chunks(pergunta, k=k)
    gen = gerar_resposta(
        pergunta,
        rec["chunks"],
        chat_history=hist_msgs,
    )

    if usar_memoria:
        hist = get_session_history(session_id)
        hist.add_message(HumanMessage(content=pergunta))
        hist.add_message(AIMessage(content=gen["resposta"]))
        trim_history(session_id)

    return {
        "indexacao": None,
        "recuperacao": rec,
        "geracao": gen,
        "resposta": gen["resposta"],
        "sources": gen.get("fontes", []),
    }
