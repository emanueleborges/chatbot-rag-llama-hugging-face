# app.py
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import uuid
import os

from auth import get_current_user, authenticate_user, create_access_token
from database import VectorDatabase
from rag_pipeline import RAGPipeline
from image_gen import ImageGenerator

app = FastAPI(title="Chatbot RAG Completo")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicialização
vector_db = VectorDatabase()
rag = RAGPipeline()
image_gen = ImageGenerator()

# Schemas
class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str
    use_cache: bool = True

class ImageRequest(BaseModel):
    prompt: str
    enhance: bool = True
    steps: int = 25

class DocumentRequest(BaseModel):
    text: str
    metadata: Optional[dict] = None

# Endpoints de Autenticação
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

# Endpoints de Documentos
@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload de documento PDF/TXT"""
    content = await file.read()
    text = content.decode("utf-8") if file.filename.endswith('.txt') else "PDF processing..."
    
    # TODO: Adicionar processamento real de PDF
    ids = vector_db.add_document(
        text,
        metadata={"filename": file.filename, "uploaded_by": current_user["sub"]}
    )
    
    return {"message": "Documento processado", "chunks": len(ids)}

@app.post("/documents/text")
async def add_text(
    request: DocumentRequest,
    current_user: dict = Depends(get_current_user)
):
    """Adiciona texto diretamente"""
    ids = vector_db.add_document(
        request.text,
        metadata=request.metadata or {"uploaded_by": current_user["sub"]}
    )
    return {"message": "Texto adicionado", "chunks": len(ids)}

# Endpoint RAG com Streaming
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """Chat com resposta em streaming"""
    
    async def generate():
        async for token in rag.query_stream(
            question=request.message,
            user_id=current_user["sub"],
            use_cache=request.use_cache
        ):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# Endpoint RAG síncrono
@app.post("/chat")
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """Chat sem streaming"""
    result = await rag.query(request.message, user_id=current_user["sub"])
    return result

# Endpoint de Geração de Imagem
@app.post("/image/generate")
async def generate_image(request: ImageRequest, current_user: dict = Depends(get_current_user)):
    """Gera imagem usando Ollama + Stable Diffusion"""
    
    image_bytes = await image_gen.generate_image(
        prompt=request.prompt,
        enhance=request.enhance,
        steps=request.steps
    )
    
    if not image_bytes:
        raise HTTPException(status_code=500, detail="Falha na geração da imagem")
    
    # Salva e retorna
    filename = f"{uuid.uuid4().hex}.png"
    filepath = image_gen.save_image(image_bytes, filename)
    
    return FileResponse(filepath, media_type="image/png", filename=filename)

# Utilitários
@app.delete("/cache")
async def clear_cache(current_user: dict = Depends(get_current_user)):
    """Limpa o cache Redis"""
    vector_db.clear_cache()
    return {"message": "Cache limpo"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)