"""
Descarrega PDFs da documentação Lenovo para knowledge/pdf_source/.

Uso responsável: indique URLs concretas (manual, guia) ou uma página de produto
onde o suporte lista PDFs. Não é um crawler do site inteiro — respeite os termos
de uso da Lenovo e o robots.txt.

Exemplos:

  python download_lenovo_pdfs.py "https://download.lenovo.com/.../manual.pdf"
  python download_lenovo_pdfs.py --page "https://pcsupport.lenovo.com/br/pt/product/..."
  python download_lenovo_pdfs.py --urls-file urls_lenovo.txt

Domínios típicos: support.lenovo.com, pcsupport.lenovo.com, download.lenovo.com
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import httpx

_ROOT = Path(__file__).resolve().parent
DEST_DIR = _ROOT / "knowledge" / "pdf_source"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Links a .pdf em atributos href/src (HTML)
_PDF_IN_ATTR = re.compile(
    r'(?:href|src)\s*=\s*["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']',
    re.IGNORECASE,
)


def _client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
        follow_redirects=True,
        timeout=httpx.Timeout(120.0, connect=30.0),
    )


def _filename_from_response(url: str, response: httpx.Response) -> str:
    cd = response.headers.get("content-disposition") or ""
    m = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';\n]+)', cd, re.I)
    if m:
        name = unquote(m.group(1).strip().strip('"'))
        if name.lower().endswith(".pdf"):
            return Path(name).name
    path = urlparse(str(response.url)).path
    base = Path(path).name
    if base.lower().endswith(".pdf"):
        return base
    slug = re.sub(r"[^\w\-]+", "_", urlparse(url).path.strip("/"))[:80] or "download"
    return f"{slug}.pdf"


def download_pdf(client: httpx.Client, url: str, dest_dir: Path, force: bool) -> Path | None:
    url = url.strip()
    if not url.lower().startswith(("http://", "https://")):
        print(f"Ignorado (não é URL http(s)): {url}", file=sys.stderr)
        return None
    try:
        r = client.get(url)
        r.raise_for_status()
    except httpx.HTTPError as e:
        print(f"Erro HTTP {url}: {e}", file=sys.stderr)
        return None

    ctype = (r.headers.get("content-type") or "").lower()
    body = r.content[:5]
    if b"%PDF-" not in body and "pdf" not in ctype:
        print(f"Aviso: resposta pode não ser PDF ({ctype[:40]}) — {url}", file=sys.stderr)

    name = _filename_from_response(url, r)
    out = dest_dir / name
    if out.exists() and not force:
        print(f"Já existe (use --force): {out.name}")
        return out

    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    tmp.write_bytes(r.content)
    tmp.replace(out)
    print(f"OK {out.name} ({len(r.content):,} bytes)")
    return out


def collect_pdf_urls_from_html(base_url: str, html: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in _PDF_IN_ATTR.finditer(html):
        href = m.group(1).strip()
        if not href or href.startswith("#") or href.lower().startswith("javascript:"):
            continue
        absolute = urljoin(base_url, href)
        if absolute.lower().endswith(".pdf") or ".pdf?" in absolute.lower():
            if absolute not in seen:
                seen.add(absolute)
                out.append(absolute)
    return out


def scrape_page(client: httpx.Client, page_url: str) -> list[str]:
    try:
        r = client.get(page_url)
        r.raise_for_status()
    except httpx.HTTPError as e:
        print(f"Erro ao abrir página: {e}", file=sys.stderr)
        return []
    enc = r.encoding or "utf-8"
    html = r.text if isinstance(r.text, str) else r.content.decode(enc, errors="replace")
    return collect_pdf_urls_from_html(str(r.url), html)


def main() -> int:
    p = argparse.ArgumentParser(description="Descarregar PDFs Lenovo para knowledge/pdf_source/")
    p.add_argument(
        "urls",
        nargs="*",
        help="URLs diretas de ficheiros .pdf",
    )
    p.add_argument(
        "--page",
        metavar="URL",
        help="Página de suporte Lenovo: extrai links .pdf e descarrega (use --max para limitar)",
    )
    p.add_argument(
        "--urls-file",
        type=Path,
        help="Ficheiro de texto com um URL por linha (# comentário permitido)",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=DEST_DIR,
        help=f"Pasta de destino (por omissão: {DEST_DIR})",
    )
    p.add_argument(
        "--max",
        type=int,
        default=20,
        help="Máximo de PDFs a descarregar quando --page (por omissão 20)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Substituir ficheiros já existentes",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Só listar URLs que seriam descarregadas (com --page)",
    )
    args = p.parse_args()

    todo: list[str] = list(args.urls)
    if args.urls_file:
        if not args.urls_file.is_file():
            print(f"Ficheiro não encontrado: {args.urls_file}", file=sys.stderr)
            return 1
        for line in args.urls_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            todo.append(line)

    page_ok = True
    with _client() as client:
        if args.page:
            found = scrape_page(client, args.page.strip())
            if not found:
                print(
                    "Nenhum link .pdf encontrado no HTML. "
                    "Abra a página no browser, use «Inspecionar» e copie o URL direto do PDF, "
                    "ou passe URLs .pdf como argumentos / --urls-file.",
                    file=sys.stderr,
                )
                page_ok = False
            else:
                print(f"Encontrados {len(found)} link(s) .pdf na página (limite --max {args.max}):")
                for u in found[: args.max]:
                    print(f"  {u}")
                if args.dry_run:
                    return 0
                for u in found[: args.max]:
                    download_pdf(client, u, args.dest, args.force)

        for u in todo:
            download_pdf(client, u, args.dest, args.force)

    if not page_ok and not todo:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
