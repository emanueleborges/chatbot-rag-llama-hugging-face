"""Refinamento: feedback do usuário → JSONL + reindexação na coleção de refinamento."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from langchain_core.documents import Document

from backend.config import FEEDBACK_JSONL
from backend.rag_engine import COLLECTION_REFINE, add_documents, index_text


def save_feedback(
    *,
    pergunta: str,
    resposta: str,
    util: bool,
    correcao: str = "",
    session_id: str = "default",
) -> None:
    FEEDBACK_JSONL.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "pergunta": pergunta,
        "resposta": resposta,
        "util": util,
        "correcao": correcao,
    }
    with FEEDBACK_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if util and correcao.strip():
        texto = f"Pergunta: {pergunta}\n\nResposta refinada:\n{correcao.strip()}"
        index_text(
            texto,
            source=f"feedback:{session_id}",
            extra_meta={"tipo": "refinamento", "util": True},
            collection=COLLECTION_REFINE,
        )
    elif util and resposta.strip():
        texto = f"Pergunta: {pergunta}\n\nResposta aprovada:\n{resposta.strip()}"
        index_text(
            texto,
            source=f"feedback:{session_id}",
            extra_meta={"tipo": "refinamento", "util": True},
            collection=COLLECTION_REFINE,
        )


def load_feedback(limit: int = 50) -> list[dict]:
    if not FEEDBACK_JSONL.is_file():
        return []
    lines = FEEDBACK_JSONL.read_text(encoding="utf-8").strip().splitlines()
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def reindex_feedback_to_chroma() -> int:
    rows = [r for r in load_feedback(500) if r.get("util")]
    docs: list[Document] = []
    for r in rows:
        cor = (r.get("correcao") or "").strip()
        resp = cor or (r.get("resposta") or "").strip()
        if not resp:
            continue
        texto = f"Pergunta: {r.get('pergunta', '')}\n\nResposta:\n{resp}"
        docs.append(
            Document(
                page_content=texto,
                metadata={
                    "source": f"feedback:{r.get('session_id', '')}",
                    "tipo": "refinamento",
                },
            )
        )
    return add_documents(docs, collection=COLLECTION_REFINE)
