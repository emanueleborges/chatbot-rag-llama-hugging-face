"""
Extrai texto de PDFs oficiais (manuais, guias de utilizador, folhas de dados)
para ficheiros .txt compatíveis com o RAG (blocos separados por linha em branco dupla).

Uso típico: descarregar PDF do suporte Lenovo/Dell (ou outro fabricante), colocar
em knowledge/pdf_source/ e gerar um .txt para revisão humana antes de fundir
em lenovo_knowledge.txt / dell_knowledge.txt.

Respeite copyright e termos de uso dos PDFs; só integre excertos que a sua
organização tenha direito de reproduzir ou usar internamente.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_KNOWLEDGE = Path(__file__).resolve().parent / "knowledge"


def _normalize(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def extract_pages(pdf_path: Path) -> list[str]:
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        print(
            "Instale PyMuPDF: pip install pymupdf",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    doc = fitz.open(pdf_path)
    pages: list[str] = []
    try:
        for i in range(len(doc)):
            raw = doc[i].get_text("text") or ""
            cleaned = _normalize(raw)
            if cleaned:
                pages.append(cleaned)
    finally:
        doc.close()
    return pages


def pdf_to_rag_chunks(
    pdf_path: Path,
    *,
    stem_label: str | None = None,
) -> str:
    """Devolve um único string com blocos separados por \\n\\n (formato RAG)."""
    label = stem_label or pdf_path.stem
    pages = extract_pages(pdf_path)
    if not pages:
        return ""

    header = (
        f"Documentação oficial em PDF — ficheiro: {pdf_path.name}\n"
        f"Texto extraído automaticamente; validar factos e remover duplicados antes de produção.\n"
    )
    blocks: list[str] = [header.strip()]
    for n, body in enumerate(pages, start=1):
        blocks.append(f"--- {label} · página {n} ---\n{body}")
    return "\n\n".join(blocks)


def main() -> int:
    p = argparse.ArgumentParser(description="Extrair texto de PDF(s) para base RAG")
    p.add_argument(
        "input_path",
        type=Path,
        help="Ficheiro .pdf ou pasta com .pdf (ex.: knowledge/pdf_source)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Ficheiro .txt de saída (ex.: knowledge/lenovo_manual_extraido.txt)",
    )
    args = p.parse_args()
    inp = args.input_path.resolve()
    out = args.output.resolve()

    if inp.is_dir():
        pdfs = sorted(inp.glob("*.pdf"))
        if not pdfs:
            print(f"Nenhum .pdf em {inp}", file=sys.stderr)
            return 1
    elif inp.suffix.lower() == ".pdf":
        pdfs = [inp]
    else:
        print("input_path deve ser .pdf ou uma pasta", file=sys.stderr)
        return 1

    parts: list[str] = []
    for pdf in pdfs:
        text = pdf_to_rag_chunks(pdf)
        if not text:
            print(f"Aviso: sem texto extraído de {pdf.name} (PDF vazio ou só imagem?)", file=sys.stderr)
            continue
        parts.append(text)

    if not parts:
        print("Nada a gravar.", file=sys.stderr)
        return 1

    merged = "\n\n".join(parts) if len(parts) > 1 else parts[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(merged, encoding="utf-8")
    print(f"Gravado: {out} ({len(merged):,} caracteres, {len(pdfs)} PDF(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
