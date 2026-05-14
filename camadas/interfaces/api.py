from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from auth import authenticate_user, create_access_token, get_current_user
from camadas.application.conversational_rag_service import ConversationalRAGService
from image_gen import ImageGenerator


@dataclass(frozen=True)
class AppComposition:
    rag_service: ConversationalRAGService
    image_generator: ImageGenerator
    clear_cache: Callable[[], None] | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    use_cache: bool = True


class ImageRequest(BaseModel):
    prompt: str
    enhance: bool = True
    steps: int = 25


class DocumentRequest(BaseModel):
    text: str
    metadata: dict | None = None


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CHAT_UI_PATH = _PROJECT_ROOT / "static" / "index.html"


def create_app(composition: AppComposition) -> FastAPI:
    app = FastAPI(title="Chatbot RAG (Clean Architecture)")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    rag = composition.rag_service
    image_gen = composition.image_generator

    @app.get("/", include_in_schema=False)
    async def chat_web_ui():
        if not _CHAT_UI_PATH.is_file():
            raise HTTPException(
                status_code=404,
                detail="Interface web não encontrada (static/index.html).",
            )
        return FileResponse(_CHAT_UI_PATH, media_type="text/html; charset=utf-8")

    @app.post("/auth/login")
    async def login(request: LoginRequest):
        user = authenticate_user(request.username, request.password)
        if not user:
            raise HTTPException(status_code=401, detail="Credenciais inválidas")
        token = create_access_token({"sub": user["username"], "role": user["role"]})
        return {"access_token": token, "token_type": "bearer"}

    @app.get("/auth/verify")
    async def verify(current_user: dict = Depends(get_current_user)):
        return {"valid": True, "user": current_user}

    @app.post("/documents/upload")
    async def upload_document(
        file: UploadFile = File(...),
        current_user: dict = Depends(get_current_user),
    ):
        content = await file.read()
        if file.filename and file.filename.lower().endswith(".txt"):
            text = content.decode("utf-8")
        else:
            text = content.decode("utf-8", errors="ignore")
        meta = {"filename": file.filename, "uploaded_by": current_user["sub"]}
        ids = rag.ingest_plain_text(text, base_metadata=meta)
        return {"message": "Documento processado", "chunks": len(ids)}

    @app.post("/documents/text")
    async def add_text(
        request: DocumentRequest,
        current_user: dict = Depends(get_current_user),
    ):
        meta = dict(request.metadata or {})
        meta.setdefault("uploaded_by", current_user["sub"])
        ids = rag.ingest_plain_text(request.text, base_metadata=meta)
        return {"message": "Texto adicionado", "chunks": len(ids)}

    @app.post("/chat/stream")
    async def chat_stream(
        request: ChatRequest,
        current_user: dict = Depends(get_current_user),
    ):
        session_id = request.session_id or str(current_user.get("sub", "anon"))

        async def generate():
            async for token in rag.converse_stream(session_id, request.message):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/chat")
    async def chat(
        request: ChatRequest,
        current_user: dict = Depends(get_current_user),
    ):
        _ = request.use_cache
        session_id = request.session_id or str(current_user.get("sub", "anon"))
        result = await rag.converse(session_id, request.message)
        return {
            "answer": result.answer,
            "standalone_question": result.standalone_question,
            "history": list(result.history),
            "sources": [
                {"text": s.text, "relevance": s.score} for s in result.sources
            ],
            "input_blocked": result.input_blocked,
            "output_blocked": result.output_blocked,
            "block_reason": result.block_reason,
        }

    @app.post("/image/generate")
    async def generate_image(
        request: ImageRequest,
        current_user: dict = Depends(get_current_user),
    ):
        _ = current_user
        image_bytes = await image_gen.generate_image(
            prompt=request.prompt,
            enhance=request.enhance,
            steps=request.steps,
        )
        if not image_bytes:
            raise HTTPException(status_code=500, detail="Falha na geração da imagem")
        filename = f"{uuid.uuid4().hex}.png"
        filepath = image_gen.save_image(image_bytes, filename)
        return FileResponse(filepath, media_type="image/png", filename=filename)

    @app.delete("/cache")
    async def clear_cache(current_user: dict = Depends(get_current_user)):
        _ = current_user
        if composition.clear_cache is None:
            return {"message": "Nenhum cache configurado para limpeza"}
        composition.clear_cache()
        return {"message": "Cache limpo"}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app
