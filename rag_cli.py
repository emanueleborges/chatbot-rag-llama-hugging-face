"""
CLI único para manutenção da base RAG (PDF → texto, Chroma, validação).

  python rag_cli.py extract [args]
  python rag_cli.py ingest [--json]
  python rag_cli.py validate [--no-clear] [--json]
  python rag_cli.py pipeline [--input ...] [--skip-ingest] [--skip-validate]

Equivalentes aos antigos extract_pdf.py / ingest_chroma.py / train_rag.py / pipeline_rag_pdf.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import EV_KNOWLEDGE_FILE
from extract_pdf import run_extract

_ROOT = Path(__file__).resolve().parent


def run_ingest(*, json_output: bool = False) -> int:
    from rag_service import clear_chunks_cache
    from vector_store import ingest_all_knowledge_txt, reset_vector_store

    try:
        report = ingest_all_knowledge_txt()
    except Exception as e:
        print(f"Erro na ingestão: {e}", file=sys.stderr)
        return 1
    reset_vector_store()
    clear_chunks_cache()
    if json_output:
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


def run_validate(*, no_clear: bool = False, json_output: bool = False) -> int:
    from rag_service import clear_chunks_cache, corpus_stats, load_chunks, retrieve

    stats = corpus_stats()
    if json_output:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0

    print("=== Base RAG (knowledge/*.txt) ===\n")
    for name, meta in stats.items():
        print(f"  {name}: {meta['chunks']} chunks, {meta['chars']:,} caracteres")
    if not stats:
        print("  (nenhum .txt encontrado em knowledge/)", file=sys.stderr)
        return 1

    samples = [
        (EV_KNOWLEDGE_FILE, "bateria recarga veículo elétrico", "Carros elétricos"),
    ]
    print("\n=== Recuperação de exemplo (top_k=4) ===\n")
    for filename, query, label in samples:
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

    if not no_clear:
        clear_chunks_cache()
        print(
            "Cache de chunks limpa (servidor FastAPI: use POST /api/rag/reload para o mesmo efeito em runtime)."
        )

    return 0


def run_pipeline(
    input_path: Path,
    output: Path,
    *,
    skip_ingest: bool = False,
    skip_validate: bool = False,
) -> int:
    code = run_extract(input_path, output)
    if code != 0:
        return code

    if not skip_ingest:
        code = run_ingest(json_output=False)
        if code != 0:
            return code

    if not skip_validate:
        code = run_validate(no_clear=False, json_output=False)
        if code != 0:
            return code

    print("\nPipeline concluído com sucesso.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(
        prog="rag_cli",
        description="Manutenção da base RAG (PDF, Chroma, validação)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- extract ---
    _knowledge = _ROOT / "knowledge"
    pe = sub.add_parser("extract", help="Extrair PDF(s) para .txt (PyMuPDF)")
    pe.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=_knowledge / "pdf_source",
        help="Ficheiro .pdf ou pasta com .pdf",
    )
    pe.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_knowledge / EV_KNOWLEDGE_FILE,
        help=f"Ficheiro .txt de saída (default: knowledge/{EV_KNOWLEDGE_FILE})",
    )

    # --- ingest ---
    pi = sub.add_parser("ingest", help="Indexar knowledge/*.txt no ChromaDB")
    pi.add_argument("--json", action="store_true", help="Imprimir só JSON")

    # --- validate ---
    pv = sub.add_parser("validate", help="Estatísticas e recuperação de exemplo")
    pv.add_argument(
        "--no-clear",
        action="store_true",
        help="Não limpar a cache de chunks",
    )
    pv.add_argument("--json", action="store_true", help="Sair com corpus em JSON")

    # --- pipeline ---
    pp = sub.add_parser(
        "pipeline",
        help="Extração PDF + ingest Chroma + validação (fluxo completo)",
    )
    pp.add_argument(
        "--input",
        type=Path,
        default=_knowledge / "pdf_source",
        dest="input_path",
        help="Ficheiro .pdf ou pasta com .pdf",
    )
    pp.add_argument(
        "--output",
        type=Path,
        default=_knowledge / EV_KNOWLEDGE_FILE,
        help=f".txt extraído (default: knowledge/{EV_KNOWLEDGE_FILE})",
    )
    pp.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Não executar ingest no Chroma",
    )
    pp.add_argument(
        "--skip-validate",
        action="store_true",
        help="Não executar validação",
    )

    args = p.parse_args(argv)

    if args.cmd == "extract":
        return run_extract(args.input_path.resolve(), args.output.resolve())
    if args.cmd == "ingest":
        return run_ingest(json_output=args.json)
    if args.cmd == "validate":
        return run_validate(no_clear=args.no_clear, json_output=args.json)
    if args.cmd == "pipeline":
        return run_pipeline(
            args.input_path.resolve(),
            args.output.resolve(),
            skip_ingest=args.skip_ingest,
            skip_validate=args.skip_validate,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
