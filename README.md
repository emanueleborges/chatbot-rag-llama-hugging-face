# RAG Chat (chatbot-rag-llama-hugging-face)

Chat web com **FastAPI**, frontend estático e **roteamento automático**: respostas com **RAG** sobre documentos Dell e sobre **Manaus** (base local), conversa geral via **LLM (Ollama)** e geração de **imagens** (Hugging Face Inference API, com fallback Pollinations).

## Funcionalidades

| Rota | Descrição |
|------|-----------|
| **RAG — Dell** | Recuperação por sobreposição de termos sobre `knowledge/dell_knowledge.txt`. |
| **RAG — Manaus** | Locais, turismo e geografia em torno de Manaus usando `knowledge/manaus_local.txt` (evita “alucinações” sobre distâncias e nomes). |
| **LLM** | Modelo configurável via API compatível OpenAI (Ollama na nuvem ou **local**). |
| **Imagem** | Pedidos explícitos do tipo “crie uma imagem…” → HF (`HF_API_TOKEN`) ou Pollinations sem token. |

O cliente recebe na resposta `route`, `route_label`, texto em `content` e, quando aplicável, `image` em Base64.

## Requisitos

- Python **3.10+** (recomendado)
- Conta/chave **Ollama** na nuvem **ou** [Ollama](https://ollama.com) instalado para API local (`OLLAMA_BASE_URL` com `localhost` / `127.0.0.1`)
- Para imagens na Hugging Face: token em [HF Settings → Access Tokens](https://huggingface.co/settings/tokens) (opcional se usar só Pollinations)

## Instalação

```bash
git clone https://github.com/emanueleborges/chatbot-rag-llama-hugging-face.git
cd chatbot-rag-llama-hugging-face
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Copie o modelo de variáveis e edite **sem commitar segredos**:

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # Linux / macOS
```

## Variáveis de ambiente (`.env`)

| Variável | Descrição |
|----------|-----------|
| `OLLAMA_API_KEY` | Chave em [ollama.com/settings/keys](https://ollama.com/settings/keys). Para API **local**, pode ficar vazio se `OLLAMA_BASE_URL` apontar para `localhost` / `127.0.0.1`. |
| `OLLAMA_BASE_URL` | Base da API OpenAI-compatível (ex.: `https://ollama.com/v1` ou `http://127.0.0.1:11434/v1`). |
| `OLLAMA_MODEL` | Nome do modelo (ex.: `gpt-oss:20b`). |
| `HF_API_TOKEN` | Token Hugging Face para Inference API de imagem. |
| `HF_IMAGE_MODEL` | ID do modelo no Hub (ex.: `runwayml/stable-diffusion-v1-5`). |
| `IMAGE_BACKEND` | `auto` (padrão: HF com token, senão Pollinations), `hf` ou `pollinations`. |

O ficheiro `.env` está no `.gitignore`. **Não** versionize tokens em notas ou ficheiros soltos na raiz do repositório.

## Como executar

```bash
python main.py
```

Abra no navegador: **http://127.0.0.1:8000**

Alternativa explícita:

```bash
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

## API HTTP

| Método | Caminho | Descrição |
|--------|---------|-----------|
| `GET` | `/` | Página do chat (HTML estático). |
| `GET` | `/api/health` | Estado do serviço. |
| `POST` | `/api/chat` | Corpo JSON: `message` (string), `history` (lista opcional de `{ "role", "content" }`, últimas mensagens). |

Exemplo mínimo:

```bash
curl -s -X POST http://127.0.0.1:8000/api/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"Olá\"}"
```

(Ficheiros estáticos em `/static`.)

## Estrutura do projeto

```
├── main.py              # Entrada: uvicorn na porta 8000
├── server.py            # FastAPI, CORS, rotas /api/* e /
├── chat_engine.py       # Orquestração: RAG / LLM / imagem
├── routing.py           # Deteção de intenção (imagem, Manaus, Dell, LLM)
├── rag_service.py       # Chunks e recuperação por termos
├── llm_service.py       # Cliente OpenAI → Ollama
├── image_service.py     # HF + fallback Pollinations
├── image_prompt_enhance.py
├── knowledge/
│   ├── dell_knowledge.txt
│   └── manaus_local.txt
├── static/              # index.html, app.js, style.css
├── requirements.txt
├── .env.example
└── README.md
```

## Base de conhecimento

- **`knowledge/dell_knowledge.txt`** — Blocos separados por linha dupla; usados pelo RAG Dell.
- **`knowledge/manaus_local.txt`** — Factos locais (shoppings, praças, distâncias indicativas); o sistema prioriza este texto para perguntas sobre Manaus quando a intenção corresponde.

## Licença e contribuições

Defina a licença do repositório conforme a política da sua organização. Para contribuir, use branches e PRs; mantenha `.env` e credenciais fora do Git.
