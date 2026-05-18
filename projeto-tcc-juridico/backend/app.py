"""
API FastAPI — chat RAG, ingestão e classificação.

Arranque (na raiz projeto-tcc-juridico/):
  pip install -r backend/requirements.txt
  python -m backend.app
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from backend.chain import chat  # noqa: E402
from backend.classifier import classificar  # noqa: E402
from backend.config import API_PORT, CORS_ORIGINS, OLLAMA_MODEL, OLLAMA_URL  # noqa: E402
from backend.ingest import extract_bytes, save_upload  # noqa: E402
from backend.memory import clear_session  # noqa: E402
from backend.rag_engine import collection_stats, index_all_knowledge, index_file  # noqa: E402
from backend.rag_pipeline import (  # noqa: E402
    executar_rag,
    gerar_resposta,
    indexar_arquivo,
    recuperar_chunks,
)
from backend.training import load_feedback, reindex_feedback_to_chroma, save_feedback  # noqa: E402
from backend.knowledge_train import status_treino, treinar_base_conhecimento  # noqa: E402
from backend.avaliacao_peticao import avaliar_peticao, formatar_relatorio_markdown  # noqa: E402

app = FastAPI(
    title="Dashboard — Análise crítica de petições",
    description="Treino da base de conhecimento + avaliação comparativa com score e melhorias.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatIn(BaseModel):
    message: str = Field(..., min_length=2)
    session_id: str = "default"
    modo: str = Field("pipeline", description="ignorado — usa pipeline de 3 fases")


class PerguntaIn(BaseModel):
    pergunta: str = Field(..., min_length=2)
    k: int = Field(5, ge=1, le=20)


class TextoIn(BaseModel):
    texto: str = Field(..., min_length=50)


class TreinoIn(BaseModel):
    reset: bool = True


class AvaliarIn(BaseModel):
    texto: str = Field(..., min_length=80)
    usar_ollama: bool = True


class FeedbackIn(BaseModel):
    pergunta: str
    resposta: str
    util: bool = True
    correcao: str = ""
    session_id: str = "default"


@app.get("/health")
def health():
    from backend.llm_local import verificar_ollama

    ollama_status = verificar_ollama()
    return {
        "status": "ok",
        "ollama_url": OLLAMA_URL,
        "ollama_model": OLLAMA_MODEL,
        "ollama": ollama_status,
        "chroma": collection_stats(),
    }


@app.post("/chat")
def post_chat(body: ChatIn):
    try:
        return chat(body.message, session_id=body.session_id, modo=body.modo)
    except Exception as e:
        raise HTTPException(503, f"Erro no chat RAG: {e}") from e


@app.post("/rag/pipeline")
def post_rag_pipeline(body: ChatIn):
    """Indexação (prévia) + recuperação + geração numa só chamada."""
    try:
        return executar_rag(body.message, session_id=body.session_id)
    except Exception as e:
        raise HTTPException(503, str(e)) from e


@app.post("/rag/recuperar")
def post_rag_recuperar(body: PerguntaIn):
    """Fase 2: pergunta → embedding → similaridade → chunks."""
    return recuperar_chunks(body.pergunta, k=body.k)


@app.post("/rag/gerar")
def post_rag_gerar(body: PerguntaIn):
    """Fase 2+3: recupera chunks e gera resposta fundamentada."""
    rec = recuperar_chunks(body.pergunta, k=body.k)
    return gerar_resposta(body.pergunta, rec["chunks"])


@app.delete("/chat/{session_id}")
def delete_chat_session(session_id: str):
    clear_session(session_id)
    return {"cleared": session_id}


@app.post("/classificar")
def post_classificar(body: TextoIn):
    return classificar(body.texto)


@app.post("/ingest/indexar")
def post_indexar():
    return index_all_knowledge()


@app.post("/ingest/upload")
async def post_upload(file: UploadFile = File(...)):
    content = await file.read()
    try:
        extract_bytes(file.filename, content)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    path = save_upload(file.filename or "upload.txt", content)
    return indexar_arquivo(path)


@app.post("/rag/indexar-pdf")
async def post_indexar_pdf(file: UploadFile = File(...)):
    """Fase 1 completa: PDF → chunks → embeddings → Chroma."""
    content = await file.read()
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Envie um ficheiro .pdf")
    path = save_upload(file.filename or "peticao.pdf", content)
    try:
        return indexar_arquivo(path)
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@app.get("/feedback")
def get_feedback():
    return {"items": load_feedback()}


@app.post("/feedback")
def post_feedback(body: FeedbackIn):
    save_feedback(
        pergunta=body.pergunta,
        resposta=body.resposta,
        util=body.util,
        correcao=body.correcao,
        session_id=body.session_id,
    )
    return {"saved": True}


@app.post("/feedback/reindexar")
def post_reindex_feedback():
    return {"chunks": reindex_feedback_to_chroma()}


@app.get("/treino/status")
def get_treino_status():
    return status_treino()


@app.post("/treino/base")
def post_treinar_base(body: TreinoIn):
    try:
        return treinar_base_conhecimento(reset=body.reset)
    except Exception as e:
        raise HTTPException(500, str(e)) from e


@app.post("/avaliar")
def post_avaliar(body: AvaliarIn):
    try:
        r = avaliar_peticao(body.texto, usar_ollama=body.usar_ollama)
        if r.get("erro"):
            raise HTTPException(400, r["erro"])
        r["relatorio_md"] = formatar_relatorio_markdown(r)
        return r
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, str(e)) from e


if __name__ == "__main__":
    import os
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    uvicorn.run(
        "backend.app:app",
        host=host,
        port=API_PORT,
        reload=host == "127.0.0.1",
    )
