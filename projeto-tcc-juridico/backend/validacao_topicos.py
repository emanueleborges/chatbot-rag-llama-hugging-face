"""Validação estrutural por tópicos jurídicos (heurística)."""
from __future__ import annotations

import re

TOPICOS = {
    "qualificacao": (
        "Qualificação das partes",
        re.compile(r"\b(autor|réu|requerente|requerido|cpf|cnpj|qualifica)\b", re.I),
    ),
    "fatos": (
        "Dos fatos",
        re.compile(r"\b(dos fatos|fatos|histórico|narra)\b", re.I),
    ),
    "fundamentacao": (
        "Fundamentação jurídica",
        re.compile(
            r"\b(fundament|direito|art\.?\s*\d|artigo\s+\d|cdc|cf/88|lei\s+\d)\b", re.I
        ),
    ),
    "pedidos": (
        "Pedidos",
        re.compile(r"\b(pede|requer|postula|dos pedidos|pedidos)\b", re.I),
    ),
    "valor_causa": (
        "Valor da causa",
        re.compile(r"\b(valor da causa|R\$\s*[\d\.]+)\b", re.I),
    ),
    "jurisprudencia": (
        "Jurisprudência / precedentes",
        re.compile(
            r"\b(jurisprud|súmula|sumula|precedente|resp|stj|stf|acórdão)\b", re.I
        ),
    ),
}


def validar_topicos(texto: str) -> dict:
    checks = []
    for chave, (titulo, pat) in TOPICOS.items():
        m = pat.search(texto)
        checks.append(
            {
                "id": chave,
                "titulo": titulo,
                "presente": bool(m),
                "evidencia": (m.group(0)[:80] if m else ""),
            }
        )
    presentes = sum(1 for c in checks if c["presente"])
    total = len(checks)
    return {
        "cobertura": round(presentes / total, 2) if total else 0.0,
        "presentes": presentes,
        "total": total,
        "topicos": checks,
        "faltando": [c for c in checks if not c["presente"]],
    }
