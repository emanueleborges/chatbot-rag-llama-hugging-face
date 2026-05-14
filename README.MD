# Chatbot RAG (LangChain, Chroma, Gemini opcional)

## Objetivos cobertos pelo código

| Objetivo | Onde está implementado |
|----------|-------------------------|
| Cadeias conversacionais com LangChain | `RunnableLambda` em `camadas/infrastructure/langchain_retrieval_runnable.py` (recuperação + re-ranking); Chroma e memória via APIs LangChain |
| Memória `ConversationBufferWindowMemory` | `camadas/infrastructure/session_memory.py` (janela configurável, padrão 5 interações) |
| Recuperação + LLM para coerência do diálogo | `camadas/application/conversational_rag_service.py` (reformulação da pergunta + contexto + resposta) |
| Re-ranking contextual | `camadas/infrastructure/cross_encoder_reranker.py` (`CrossEncoderDocumentReranker`; `IdentityDocumentReranker` se `DISABLE_RERANKER=true`) |
| Guardrails e PII na saída | `camadas/infrastructure/basic_guardrails.py` (entrada/saída + redação de e-mail, CPF formatado, telefone BR, cartão 16 dígitos) |
| VectorStore ChromaDB | `camadas/infrastructure/vector_store.py` |
| Embeddings Google Generative AI | `camadas/infrastructure/gemini_embeddings.py` (quando `GEMINI_API_KEY` ou `GOOGLE_API_KEY` está definida; senão HF/OpenAI via `openai_embeddings.py`) |
| Chunking de textos | `camadas/application/rag_chat_service.py` (`_chunk_text`, usado na ingestão em `ConversationalRAGService`) |

## Fluxo resumido

1. Autenticação (JWT) → token  
2. Upload / texto → **chunking** com sobreposição  
3. **Embeddings** (Gemini ou locais) → **Chroma** persistido  
4. Chat: **memória em janela** → **reformulação** da pergunta → **Runnable** busca + **re-ranking** → **LLM** (Gemini ou Ollama) → **guardrails** (incl. redação PII na resposta)  
5. (Opcional) Geração de imagem conforme `image_gen.py`

## Variáveis de ambiente

Ver `.env.example` (`GEMINI_*`, `CHROMA_*`, `RAG_FETCH_K`, `RAG_FINAL_K`, `CONV_BUFFER_WINDOW`, etc.).

## Iniciar

```text
docker-compose up -d     # Redis + Ollama (se usar stack local)
python app.py            # ou: python main.py
```

**Interface web (estilo chat escuro):** abre no browser `http://localhost:8000/` (login JWT na barra lateral, streaming SSE, upload `.txt`).

## Exemplo de cliente (httpx)

```python
import httpx

# 1. Login
response = httpx.post(
    "http://localhost:8000/auth/login",
    json={"username": "admin", "password": "admin123"},
)
token = response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. Upload de documento
with open("documento.txt", "rb") as f:
    httpx.post(
        "http://localhost:8000/documents/upload",
        headers=headers,
        files={"file": f},
    )

# 3. Chat (histórico por sessão + reformulação + fontes na resposta JSON)
resp = httpx.post(
    "http://localhost:8000/chat",
    headers=headers,
    json={
        "message": "O que o documento diz sobre X?",
        "session_id": "minha-sessao-1",  # opcional; default: sub do JWT
    },
)
print(resp.json()["answer"])
print(resp.json().get("standalone_question"))
print(resp.json().get("history"))

# 4. Chat com streaming (SSE)
with httpx.Client() as client:
    with client.stream(
        "POST",
        "http://localhost:8000/chat/stream",
        headers=headers,
        json={"message": "Resuma em uma frase.", "session_id": "minha-sessao-1"},
    ) as stream:
        for line in stream.iter_lines():
            if line and line.startswith("data: ") and "[DONE]" not in line:
                print(line[6:], end="")

# 5. Gerar imagem
img = httpx.post(
    "http://localhost:8000/image/generate",
    headers=headers,
    json={"prompt": "um gato astronauta no espaço", "enhance": True},
)
with open("imagem.png", "wb") as f:
    f.write(img.content)
```

## Testes rápidos

```text
curl http://localhost:8000/health
curl -X POST http://localhost:8000/auth/login -H "Content-Type: application/json" -d "{\"username\":\"admin\",\"password\":\"admin123\"}"
```

## Limpar cache (pipeline legado Redis)

```text
curl -X DELETE http://localhost:8000/cache -H "Authorization: Bearer SEU_TOKEN"
```

Requer `ENABLE_LEGACY_REDIS_CACHE_CLEAR=true` no `.env` para ter efeito.

## Requisitos atendidos (resumo)

- ChromaDB persistente para índice e busca  
- Embeddings: **Google Generative AI** (com chave) ou **Sentence Transformers** / OpenAI conforme configuração  
- LLM: **Gemini** (com chave) ou **Ollama** local  
- Chunking com sobreposição  
- Memória conversacional em janela, reformulação de pergunta, re-ranking e guardrails  
- Streaming SSE (com saída filtrada antes de enviar ao cliente)  
- Autenticação JWT  
- Geração de imagem (Ollama + Stable Diffusion) quando configurado  
