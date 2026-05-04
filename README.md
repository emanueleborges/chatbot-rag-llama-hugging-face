# RAG Chat (chatbot-rag-llama-hugging-face)

Chat web com **FastAPI**, frontend estático e **roteamento automático**: respostas com **RAG** sobre documentos **Dell** e **Lenovo**, sobre **Manaus** (base local), conversa geral via **LLM (Ollama)** e geração de **imagens** (Hugging Face Inference API, com fallback Pollinations).

## Funcionalidades

| Rota | Descrição |
|------|-----------|
| **RAG — Dell** | Contexto de `knowledge/dell_knowledge.txt`: **Chroma** (similaridade semântica) se a coleção estiver indexada; caso contrário, sobreposição de termos. |
| **RAG — Lenovo** | Idem sobre `knowledge/lenovo_knowledge.txt` (ThinkPad, Legion, etc.). Ativado por palavras-chave Lenovo. |
| **RAG — Manaus** | Locais, turismo e geografia em torno de Manaus usando `knowledge/manaus_local.txt` (evita “alucinações” sobre distâncias e nomes). |
| **LLM** | Modelo configurável via API compatível OpenAI (Ollama na nuvem ou **local**). |
| **Imagem** | Pedidos explícitos do tipo “crie uma imagem…” → HF (`HF_API_TOKEN`) ou Pollinations sem token. |

O cliente recebe na resposta `route`, `route_label`, texto em `content` e, quando aplicável, `image` em Base64.

### Treino e recarga da base RAG

A «base» são os textos em `knowledge/*.txt` (chunks separados por **linha em branco dupla**). A recuperação para **Dell** e **Lenovo** usa **ChromaDB** com embeddings multilingues quando a coleção existe; se Chroma estiver vazio ou as dependências não estiverem instaladas, usa-se **sobreposição de palavras** (`rag_service.retrieve`).

- **Atualizar o conhecimento:** edite ou acrescente ficheiros em `knowledge/`, mantendo o formato de chunks.
- **Indexar no Chroma:** `python ingest_chroma.py` (grava em `chroma_db/` por omissão; primeira execução pode descarregar o modelo Sentence-Transformers).
- **Validar localmente:** `python train_rag.py` (estatísticas + exemplos de recuperação por termos).
- **Com o servidor a correr:** `POST /api/rag/reload` limpa a cache de `.txt`; `POST /api/rag/ingest-chroma` reindexa tudo no Chroma (assíncrono no worker; pode demorar).

### PDFs da documentação oficial

Para usar **material extraído de PDFs** (manuais do utilizador, guias de serviço, etc.):

1. Descarregue os PDFs a partir do **site oficial** do fabricante (ex.: [Suporte Lenovo BR](https://support.lenovo.com/br/pt), suporte Dell).
2. Coloque-os em `knowledge/pdf_source/` ou use **`python download_lenovo_pdfs.py "URL_DO_PDF"`** (e variantes `--page`, `--urls-file`; ver `knowledge/pdf_source/LEIA-ME.txt`). Não há descarga automática de «todo o site» — use URLs concretos dos manuais.
3. Execute: `python extract_pdf.py knowledge/pdf_source -o knowledge/extraido_oficial.txt`  
   Ou um único ficheiro: `python extract_pdf.py caminho/manual.pdf -o knowledge/lenovo_pdf.txt`
4. **Revise** o texto (PDFs com colunas ou páginas só-imagem saem incompletos; não há OCR neste script).
5. Integre no RAG: copie trechos validados para `lenovo_knowledge.txt` / `dell_knowledge.txt`, ou mantenha um `.txt` extra em `knowledge/` e amplie o `chat_engine` / roteamento se quiser uma rota dedicada.

Dependência: **PyMuPDF** (`pymupdf` no `requirements.txt`). Respeite copyright e termos de uso dos documentos.

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
| `CHROMA_PATH` | (Opcional) Pasta absoluta para persistência Chroma; por omissão `chroma_db/` na raiz do projeto. |
| `CHROMA_EMBED_MODEL` | (Opcional) Nome do modelo Sentence-Transformers (por omissão `paraphrase-multilingual-MiniLM-L12-v2`). |

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

## Docker

Requisito: [Docker](https://docs.docker.com/get-docker/) e Docker Compose v2.

1. Copie variáveis: `copy .env.example .env` (Windows) ou `cp .env.example .env` e preencha pelo menos `OLLAMA_*` (nuvem) ou aponte para Ollama no **host** (ver abaixo).
2. Na raiz do projeto:

```bash
docker compose up --build
```

3. Abra **http://localhost:8000**

O compose monta `./knowledge` no contentor (pode editar `.txt` no host), define `CHROMA_PATH=/data/chroma_db` com volume nomeado, e cache Hugging Face para embeddings. Está configurado `host.docker.internal` para falar com serviços no PC (ex.: Ollama local).

**Ollama a correr no Windows/Mac (fora do Docker):** no `.env` dentro do projecto (usado pelo compose):

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
OLLAMA_API_KEY=ollama
```

**Chroma dentro do contentor:** após alterar `knowledge/*.txt`, chame `POST http://localhost:8000/api/rag/ingest-chroma` ou execute ingest localmente com a mesma pasta `knowledge/`.

A imagem é grande (PyTorch + Chroma + dependências). O primeiro `docker compose build` pode demorar vários minutos.

Ficheiros: `Dockerfile`, `docker-compose.yml`, `.dockerignore`.

## API HTTP

| Método | Caminho | Descrição |
|--------|---------|-----------|
| `GET` | `/` | Página do chat (HTML estático). |
| `GET` | `/api/health` | Estado do serviço; inclui `chroma_chunks` quando Chroma está disponível e indexado. |
| `POST` | `/api/chat` | Corpo JSON: `message` (string), `history` (lista opcional de `{ "role", "content" }`, últimas mensagens). |
| `POST` | `/api/rag/reload` | Limpa cache de chunks e devolve `corpus` + `chroma_chunks` (contagem na coleção, se existir). |
| `POST` | `/api/rag/ingest-chroma` | Apaga e recria a coleção Chroma a partir de todos os `knowledge/*.txt`. |

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
├── routing.py           # Deteção de intenção (imagem, Manaus, Lenovo, Dell, LLM)
├── rag_service.py       # Chunks, cache, recuperação híbrida (Chroma + termos)
├── vector_store.py      # Chroma persistente + ingestão
├── ingest_chroma.py     # CLI: indexar knowledge/*.txt no Chroma
├── train_rag.py         # Validação da base RAG (CLI)
├── extract_pdf.py       # PDF oficial → .txt (chunks por página)
├── download_lenovo_pdfs.py  # PDFs → knowledge/pdf_source/ (URLs ou --page)
├── llm_service.py       # Cliente OpenAI → Ollama
├── image_service.py     # HF + fallback Pollinations
├── image_prompt_enhance.py
├── knowledge/
│   ├── pdf_source/      # PDFs locais (ver LEIA-ME.txt); *.pdf ignorados pelo Git
│   ├── dell_knowledge.txt
│   ├── lenovo_knowledge.txt
│   └── manaus_local.txt
├── static/              # index.html, app.js, style.css
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## Base de conhecimento

- **`knowledge/dell_knowledge.txt`** — Blocos separados por linha dupla; usados pelo RAG Dell.
- **`knowledge/lenovo_knowledge.txt`** — Documentação interna de referência (linhas de produto, suporte Brasil); usados pelo RAG Lenovo. Pode editar ou substituir por conteúdo oficial da sua organização.
- **`knowledge/manaus_local.txt`** — Factos locais (shoppings, praças, distâncias indicativas); o sistema prioriza este texto para perguntas sobre Manaus quando a intenção corresponde.

**Prioridade do roteador:** imagem → Manaus (local) → Lenovo → Dell → LLM geral. Se a mesma mensagem citar Dell e Lenovo, ganha **Lenovo** (é avaliado antes de Dell no código).

## Licença e contribuições

Defina a licença do repositório conforme a política da sua organização. Para contribuir, use branches e PRs; mantenha `.env` e credenciais fora do Git.
