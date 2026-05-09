"""Pesquisa web usada apenas quando o RAG documental não devolve trechos relevantes."""
from __future__ import annotations


def search_web_snippets(
    query: str,
    *,
    max_results: int = 8,
    region: str = "br-pt",
) -> str:
    """
    Trechos (título, resumo, URL) para o prompt do LLM.
    String vazia se falhar ou sem pacote / rede.
    """
    q = (query or "").strip()
    if not q:
        return ""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # legado
        except ImportError:
            return ""

    lines: list[str] = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(q, region=region, max_results=max_results)
            for i, r in enumerate(results, start=1):
                title = (r.get("title") or "").strip()
                body = (r.get("body") or "").strip()
                href = (r.get("href") or "").strip()
                if not (title or body):
                    continue
                lines.append(f"{i}. **{title}**\n   {body}\n   {href}")
        return "\n\n".join(lines)
    except Exception:
        return ""
