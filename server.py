"""RAG Chat — FastAPI + frontend estático."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chat_engine import process_chat
from rag_service import clear_chunks_cache, corpus_stats

STATIC = Path(__file__).resolve().parent / "static"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = ""
    history: list[ChatMessage] = Field(default_factory=list)


app = FastAPI(title="RAG Chat", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    out: dict = {"status": "ok", "app": "RAG Chat"}
    try:
        from vector_store import chroma_count

        n = chroma_count()
        if n is not None:
            out["chroma_chunks"] = n
    except ImportError:
        out["chroma_chunks"] = None
    except Exception:
        out["chroma_chunks"] = None
    return out


@app.post("/api/rag/reload")
def rag_reload():
    """Limpa cache de chunks e devolve estatísticas da pasta knowledge/ (após editar .txt)."""
    clear_chunks_cache()
    payload: dict = {"ok": True, "corpus": corpus_stats()}
    try:
        from vector_store import chroma_count

        payload["chroma_chunks"] = chroma_count()
    except Exception:
        payload["chroma_chunks"] = None
    return payload


@app.post("/api/rag/ingest-chroma")
async def rag_ingest_chroma():
    """Reindexa todos os knowledge/*.txt no Chroma (pode demorar na 1ª vez — download do modelo)."""
    import asyncio

    try:
        from vector_store import ingest_all_knowledge_txt, reset_vector_store
    except ImportError:
        return {
            "ok": False,
            "error": "Dependências Chroma em falta. Execute: pip install -r requirements.txt",
        }
    try:
        report = await asyncio.to_thread(ingest_all_knowledge_txt)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    reset_vector_store()
    clear_chunks_cache()
    return {"ok": True, **report}


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    prior = [{"role": m.role, "content": m.content} for m in req.history[-24:]]
    return await process_chat(req.message, prior)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
