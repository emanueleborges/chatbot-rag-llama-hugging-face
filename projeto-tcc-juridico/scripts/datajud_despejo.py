#!/usr/bin/env python3
"""
Busca processos de DESPEJO via API pública DataJud (CNJ).

IMPORTANTE — o DataJud NÃO fornece o texto da petição inicial, apenas metadados
(número, classe, órgão, assuntos, datas, movimentações em alguns tribunais).
Para treinar RAG com texto completo, use petições anonimizadas de escritório
ou modelos públicos em data/knowledge/.

Chave: https://www.cnj.jus.br/sistemas/datajud/api-publica/
  → solicitar API Key → export DATAJUD_API_KEY=...

Uso:
  cd projeto-tcc-juridico
  export DATAJUD_API_KEY=sua-chave
  python scripts/datajud_despejo.py --tipo falta_pagamento --size 10
  python scripts/datajud_despejo.py --tipo todos --tribunal tjsp --size 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import urllib.error
import urllib.request

BASE = "https://api-publica.datajud.cnj.jus.br"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "datajud"

# Códigos CNJ (classe processual) — conferir tabela CNJ se necessário
CLASSES = {
    "falta_pagamento": [93, 94],  # Despejo por Falta de Pagamento (+ cumulado cobrança)
    "despejo_generico": [92],
    "denuncia_vazia": [92],  # filtrar também por assunto 9612 na resposta
    "uso_proprio": [92],  # Despejo (filtrar por assunto 9610)
}

# Assuntos CNJ (complemento quando classe é genérica "Despejo")
ASSUNTOS = {
    "denuncia_vazia": [9612],
    "falta_pagamento": [14915],  # Despejo por Inadimplemento
    "uso_proprio": [9610],  # Despejo para Uso Próprio
}

TRIBUNAIS = [
    "tjsp",
    "tjrj",
    "tjmg",
    "tjrs",
    "tjpr",
    "tjsc",
    "tjba",
    "tjpe",
    "tjdf",
    "trf1",
    "trf2",
    "trf3",
    "trf4",
    "trf5",
    "trf6",
]


def _headers() -> dict:
    key = (os.getenv("DATAJUD_API_KEY") or "").strip()
    if not key:
        print(
            "Defina DATAJUD_API_KEY (chave pública DataJud no portal CNJ).",
            file=sys.stderr,
        )
        sys.exit(1)
    return {
        "Authorization": f"APIKey {key}",
        "Content-Type": "application/json",
    }


def _post(alias: str, body: dict) -> dict:
    url = f"{BASE}/{alias}/_search"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {alias}: {err[:500]}") from e


def build_query(
    tipo: str,
    *,
    size: int,
    search_after: list | None = None,
) -> dict:
    must = [{"terms": {"classe.codigo": CLASSES.get(tipo, CLASSES["falta_pagamento"])}}]
    if tipo in ASSUNTOS:
        must.append({"terms": {"assuntos.codigo": ASSUNTOS[tipo]}})

    body: dict = {
        "size": size,
        "query": {
            "bool": {
                "must": must,
                "must_not": [{"range": {"nivelSigilo": {"gt": 0}}}],
            }
        },
        "sort": [{"dataAjuizamento": {"order": "desc"}}, {"id.keyword": "asc"}],
    }
    if search_after:
        body["search_after"] = search_after
    return body


def parse_hit(hit: dict, tribunal: str) -> dict:
    src = hit.get("_source", {})
    classe = src.get("classe") or {}
    orgao = src.get("orgaoJulgador") or {}
    assuntos = src.get("assuntos") or []
    return {
        "numero_processo": src.get("numeroProcesso"),
        "classe_codigo": classe.get("codigo"),
        "classe_nome": classe.get("nome"),
        "grau": src.get("grau"),
        "data_ajuizamento": src.get("dataAjuizamento"),
        "nivel_sigilo": src.get("nivelSigilo", 0),
        "orgao_codigo": orgao.get("codigo"),
        "orgao_nome": orgao.get("nome"),
        "assuntos": [
            {"codigo": a.get("codigo"), "nome": a.get("nome")} for a in assuntos
        ],
        "tribunal": tribunal,
        "score": hit.get("_sort"),
    }


def buscar_tribunal(alias: str, tipo: str, size: int) -> dict:
    body = build_query(tipo, size=size)
    raw = _post(f"api_publica_{alias}", body)
    hits = raw.get("hits", {})
    total = hits.get("total", {})
    if isinstance(total, dict):
        total_str = total.get("value", 0)
        rel = total.get("relation", "eq")
        total_aprox = f"{total_str}+" if rel == "gte" else str(total_str)
    else:
        total_aprox = str(total)
    processos = [parse_hit(h, alias.replace("api_publica_", "")) for h in hits.get("hits", [])]
    return {
        "total_aproximado": total_aprox,
        "retornados": len(processos),
        "processos": processos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Busca despejo no DataJud (metadados).")
    parser.add_argument(
        "--tipo",
        choices=["falta_pagamento", "despejo_generico", "denuncia_vazia", "uso_proprio", "todos"],
        default="falta_pagamento",
    )
    parser.add_argument("--tribunal", default="", help="Ex.: tjsp (vazio = vários TJs)")
    parser.add_argument("--size", type=int, default=10, help="Processos por tribunal")
    parser.add_argument("-o", "--output", default="", help="Ficheiro JSON de saída")
    args = parser.parse_args()

    tribunais = [args.tribunal] if args.tribunal else TRIBUNAIS[:6]
    tipos = (
        ["falta_pagamento", "denuncia_vazia", "uso_proprio"]
        if args.tipo == "todos"
        else [args.tipo]
    )

    resultado = {"tipos": tipos, "size_por_tribunal": args.size, "tribunais": {}}
    for alias in tribunais:
        alias = alias if alias.startswith("api_publica_") else f"api_publica_{alias}"
        tribunal_key = alias.replace("api_publica_", "")
        resultado["tribunais"][tribunal_key] = {}
        for tipo in tipos:
            try:
                resultado["tribunais"][tribunal_key][tipo] = buscar_tribunal(
                    tribunal_key, tipo, args.size
                )
                time.sleep(0.5)
            except Exception as e:
                resultado["tribunais"][tribunal_key][tipo] = {"erro": str(e)}

    resultado["aviso"] = (
        "Metadados públicos DataJud — SEM texto de petição inicial. "
        "Não redistribuir processos sigilosos. Para RAG: anonimizar PDFs próprios."
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else OUT_DIR / f"despejo_{args.tipo}.json"
    out.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Gravado: {out} ({sum(len(v.get(t, {}).get('processos', [])) for v in resultado['tribunais'].values() for t in tipos if isinstance(v.get(t), dict))} processos)")


if __name__ == "__main__":
    main()
