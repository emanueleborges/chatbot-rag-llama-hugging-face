# Exemplo de cliente
import httpx
import json

# 1. Login
response = httpx.post("http://localhost:8000/auth/login", json={
    "username": "admin",
    "password": "admin123"
})
token = response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. Upload de documento
with open("documento.txt", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/documents/upload",
        headers=headers,
        files={"file": f}
    )

# 3. Chat com streaming
async with httpx.AsyncClient() as client:
    async with client.stream(
        "POST",
        "http://localhost:8000/chat/stream",
        headers=headers,
        json={"message": "O que o documento diz sobre X?"}
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                token = line[6:]
                if token != "[DONE]":
                    print(token, end="")

# 4. Gerar imagem
response = httpx.post(
    "http://localhost:8000/image/generate",
    headers=headers,
    json={"prompt": "um gato astronauta no espaço", "enhance": True}
)

# Salva imagem
with open("imagem.png", "wb") as f:
    f.write(response.content)



    1. Autenticação (JWT) → Token
                ↓
2. Upload Documento → Chunking (sobreposição 20%)
                ↓
3. Embedding (SentenceTransformer/Ollama)
                ↓
4. ChromaDB (persistente) + Redis (cache)
                ↓
5. Query → Cache? → Busca → Contexto
                ↓
6. Ollama (llama3.2) → Streaming → Resposta
                ↓
7. Geração de Imagem → Ollama (prompt enhance) → Stable Diffusion


# Iniciar tudo
docker-compose up -d     # Redis + Ollama
python app.py            # API Principal

# Testar endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/auth/login -H "Content-Type: application/json" -d '{"username":"admin","password":"admin123"}'

# Limpar cache
curl -X DELETE http://localhost:8000/cache -H "Authorization: Bearer SEU_TOKEN"


Esta implementação inclui todos os requisitos que você pediu:

✅ Banco de dados (ChromaDB persistente)

✅ Embeddings (SentenceTransformer + Ollama)

✅ LLM Ollama local (llama3.2:3b)

✅ Criação de imagem (Ollama + Stable Diffusion)

✅ Chunks com sobreposição

✅ Cache com Redis

✅ Streaming de respostas (SSE)

✅ Autenticação JWT