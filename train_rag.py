"""
Validação e «treino» da base RAG (ficheiros em knowledge/).

Este projeto não treina embeddings nem rede neural: o contexto vem dos .txt
separados por linha em branco dupla; a recuperação é por sobreposição de termos.
Este script confirma que os ficheiros existem, mede chunks e simula consultas.
"""
from __future__ import annotations

import argparse
import json
import sys

from rag_service import clear_chunks_cache, corpus_stats, load_chunks, retrieve


def _samples() -> list[tuple[str, str, str]]:
    """(ficheiro, pergunta de teste, etiqueta)."""
    return [
        ("dell_knowledge.txt", "garantia e service tag no Brasil", "Dell"),
        ("lenovo_knowledge.txt", "ThinkPad TPM BIOS drivers MTM", "Lenovo"),
        ("manaus_local.txt", "shopping praça manaus", "Manaus"),
    ]


def main() -> int:
    p = argparse.ArgumentParser(description="Validar base RAG em knowledge/")
    p.add_argument(
        "--no-clear",
        action="store_true",
        help="Não limpar a cache (útil só para inspeção em processo isolado)",
    )
    p.add_argument("--json", action="store_true", help="Sair com corpus em JSON")
    args = p.parse_args()

    stats = corpus_stats()
    if args.json:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0

    print("=== Base RAG (knowledge/*.txt) ===\n")
    for name, meta in stats.items():
        print(f"  {name}: {meta['chunks']} chunks, {meta['chars']:,} caracteres")
    if not stats:
        print("  (nenhum .txt encontrado em knowledge/)", file=sys.stderr)
        return 1

    print("\n=== Recuperação de exemplo (top_k=4) ===\n")
    for filename, query, label in _samples():
        if filename not in stats:
            print(f"  [{label}] ficheiro em falta: {filename}\n")
            continue
        chunks = load_chunks(filename)
        picked = retrieve(query, chunks, top_k=4)
        print(f"  [{label}] {filename}")
        print(f"      pergunta: {query!r}")
        print(f"      trechos: {len(picked)}")
        for i, block in enumerate(picked, 1):
            head = block.replace("\n", " ")[:120]
            print(f"      --- {i}: {head}...")
        print()

    if not args.no_clear:
        clear_chunks_cache()
        print("Cache de chunks limpa (servidor FastAPI: use POST /api/rag/reload para o mesmo efeito em runtime).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
