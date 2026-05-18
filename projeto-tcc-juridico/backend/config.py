"""Configuração central do projeto."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
UPLOADS_DIR = DATA_DIR / "uploads"
EXEMPLOS_JSON = DATA_DIR / "peticoes_exemplo.json"
FEEDBACK_JSONL = DATA_DIR / "feedback.jsonl"
MODELS_DIR = ROOT / "models" / "classificador_ruim_bom"
CHROMA_DIR = str(ROOT / "chroma_db")

# Chunk size adaptativo: 1200 para petições longas, 900 para curtas
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
COLLECTION_DOCS = "juridico_docs"
COLLECTION_REFINE = "juridico_refinamento"
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
# Chunks recuperados na comparação (mais alto = mais petições de referência)
COMPARACAO_TOP_K = int(os.getenv("COMPARACAO_TOP_K", "12"))

_raw_url = (os.getenv("OLLAMA_URL") or "http://127.0.0.1:11434").strip().rstrip("/")
OLLAMA_URL = _raw_url.replace("://localhost:", "://127.0.0.1:")
OLLAMA_MODEL = (os.getenv("OLLAMA_MODEL") or "llama3.2:3b").strip()
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))

MEMORY_WINDOW = int(os.getenv("CONV_BUFFER_WINDOW", "6"))
API_PORT = int(os.getenv("JURIDICO_API_PORT", "8011"))
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7861"))

# CORS — permitir origens específicas em produção
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

LABELS = ["DEU RUIM", "DEU BOM", "MEIO CERTO"]

# Tipos de documentos para classificação semântica
TIPOS_DOCUMENTO = {
    "peticao_inicial": ["petição", "peticao", "excelentíssimo", "excelentissimo", "requer", "pede"],
    "sentenca": ["sentença", "sentenca", "julgo", "procedente", "improcedente"],
    "acordao": ["acórdão", "acordao", "apelação", "apelacao", "recurso", "relator"],
    "decisao": ["decisão", "decisao", "despacho", "defiro", "indefiro"],
    "outros": [],
}

for d in (KNOWLEDGE_DIR, UPLOADS_DIR, MODELS_DIR, Path(CHROMA_DIR)):
    d.mkdir(parents=True, exist_ok=True)
