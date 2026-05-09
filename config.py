"""Constantes e limites de ambiente — único lugar para nomes de ficheiros e quotas.

Evita strings mágicas espalhadas e facilita trocar a base RAG ou limites sem mexer na lógica.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Ficheiro único da base documental em knowledge/
EV_KNOWLEDGE_FILE = "ev_knowledge.txt"

# Chroma indexa apenas estes nomes (alinhado ao RAG)
CHROMA_SOURCE_FILES: tuple[str, ...] = (EV_KNOWLEDGE_FILE,)


def max_pdf_upload_bytes() -> int:
    mb = int(os.environ.get("MAX_PDF_UPLOAD_MB", "25"))
    return mb * 1024 * 1024


def rag_strict_grounding() -> bool:
    """Se True, só responde com trechos recuperados; sem fallback web/LLM geral."""
    return os.environ.get("RAG_STRICT_GROUNDING", "").strip().lower() in ("1", "true", "yes")


def rag_chroma_max_distance() -> float | None:
    """
    Distância máxima Chroma (menor = mais similar; escala depende do espaço de embeddings).
    Se definido, descarta trechos menos similares à consulta. None = sem filtro por distância.
    """
    raw = os.environ.get("RAG_CHROMA_MAX_DISTANCE", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def rag_lexical_min_overlap() -> int:
    """Mínimo de palavras significativas em comum (pergunta ∩ trecho) para aceitar fallback lexical."""
    raw = os.environ.get("RAG_LEXICAL_MIN_OVERLAP", "1").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def chat_map_geocode_enabled() -> bool:
    """Se True, tenta geocodificar perguntas com indício de local e devolve `map` na API."""
    return os.environ.get("ENABLE_CHAT_MAP", "1").strip().lower() not in ("0", "false", "no")


def openchargemap_api_key() -> str:
    """Chave gratuita em https://openchargemap.org/site/develop/api — necessária para marcar postos no mapa."""
    return os.environ.get("OPENCHARGEMAP_API_KEY", "").strip()


def openchargemap_radius_km() -> int:
    raw = os.environ.get("OPENCHARGEMAP_RADIUS_KM", "35").strip()
    try:
        return max(5, min(100, int(raw)))
    except ValueError:
        return 35


def openchargemap_max_results() -> int:
    raw = os.environ.get("OPENCHARGEMAP_MAX_RESULTS", "40").strip()
    try:
        return max(5, min(80, int(raw)))
    except ValueError:
        return 40
