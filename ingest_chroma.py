"""
Ingere todos os `knowledge/*.txt` no Chroma (embeddings + metadado `source`).
Execute após alterar a base textual ou na primeira configuração.

  pip install -r requirements.txt
  python ingest_chroma.py

Variáveis opcionais: CHROMA_PATH, CHROMA_EMBED_MODEL (ver .env.example).
"""
from __future__ import annotations

import json
import sys

from rag_service import clear_chunks_cache
from vector_store import ingest_all_knowledge_txt, reset_vector_store


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Ingere knowledge/*.txt no ChromaDB")
    p.add_argument("--json", action="store_true", help="Imprimir só JSON")
    args = p.parse_args()
    try:
        report = ingest_all_knowledge_txt()
    except Exception as e:
        print(f"Erro na ingestão: {e}", file=sys.stderr)
        return 1
    reset_vector_store()
    clear_chunks_cache()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("Chroma ingestão concluída.")
        print(f"  Pasta: {report['chroma_path']}")
        print(f"  Coleção: {report['collection']}")
        print(f"  Modelo de embeddings: {report['embed_model']}")
        print(f"  Total de chunks: {report['total_chunks']}")
        for k, v in report["chunks_by_file"].items():
            print(f"    - {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
