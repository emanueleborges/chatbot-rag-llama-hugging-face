"""Avaliação de uma petição face à base treinada (score + melhorias)."""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from collections import defaultdict

from backend.classifier import classificar
from backend.config import COMPARACAO_TOP_K, RAG_TOP_K
from backend.rag_pipeline import recuperar_chunks
from backend.validacao_topicos import TOPICOS, validar_topicos

_LABEL_SCORE = {"DEU BOM": 92, "MEIO CERTO": 62, "DEU RUIM": 35}

_PROMPT_AVALIACAO = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Você é perito em análise comparativa de petições jurídicas.
A BASE DE CONHECIMENTO contém petições já treinadas (modelos). Compare a PETIÇÃO DO USUÁRIO
com CADA UMA dessas referências: o que a petição do usuário tem de igual, o que falta em
relação aos modelos, e o que está mais fraco.
Cite os nomes dos ficheiros de referência quando relevante. Não invente fatos nem jurisprudência.
Responda APENAS em JSON válido (sem markdown, sem ```):
{{
  "pontos_fortes": ["..."],
  "analise_critica": "comparação directa: sua petição vs petições treinadas (cite documentos)",
  "comparacao_por_referencia": [
    {{"documento": "nome.pdf", "semelhanca": "alta|media|baixa", "o_que_falta_na_sua": "...", "o_que_esta_bem": "..."}}
  ],
  "melhorias": [
    {{"secao": "secção", "problema": "em relação às petições treinadas, o que falta", "sugestao": "ação concreta", "prioridade": "alta|media|baixa", "referencia_modelo": "ficheiro.pdf"}}
  ],
  OBRIGATÓRIO: inclua pelo menos 3 itens em "melhorias", mesmo que a petição seja boa (sugira refinamentos).
  "resumo": "síntese da comparação em 2-3 frases"
}}""",
        ),
        (
            "human",
            """PETIÇÕES JÁ TREINADAS (referência — trechos mais próximos):
{exemplos}

RANKING DE SIMILARIDADE POR DOCUMENTO:
{ranking_docs}

---
PETIÇÃO DO USUÁRIO (a comparar):
{peticao}

Tópicos estruturais em falta na petição do usuário: {faltando}
Classificação heurística da petição do usuário: {label}
""",
        ),
    ]
)

_PROMPT_COMPLETUDE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Você é um avaliador de petições jurídicas. Analise a completude da petição abaixo.
Responda APENAS em JSON válido (sem markdown, sem ```):
{{
  "completude_geral": "baixa|media|alta",
  "pontuacao": 0-100,
  "observacoes": ["..."],
  "sugestoes": ["..."]
}}
Considere: qualificação das partes, exposição dos fatos, fundamentação jurídica com artigos de lei,
pedidos claros e específicos, valor da causa, jurisprudência ou súmulas."""
        ),
        ("human", "PETIÇÃO:\n{peticao}"),
    ]
)


def _score_estrutura(texto: str) -> tuple[float, dict]:
    v = validar_topicos(texto)
    return v["cobertura"] * 100, v


def _score_classificacao(texto: str) -> tuple[float, dict]:
    c = classificar(texto)
    base = _LABEL_SCORE.get(c["label"], 50)
    cob = c.get("cobertura_topicos")
    if cob is not None:
        base = min(100, base * 0.7 + float(cob) * 100 * 0.3)
    return float(base), c


def _nome_documento(source: str) -> str:
    s = (source or "").replace("\\", "/")
    return s.split("/")[-1] if s else "desconhecido"


def _aspas_num(v) -> str:
    """Quantidade apenas entre aspas, ex.: \"78\"."""
    if v is None:
        return '"—"'
    if isinstance(v, (int, float)):
        n = float(v)
        return f'"{int(round(n))}"' if n == int(n) else f'"{n:.1f}"'
    s = str(v).strip().rstrip("%")
    if s.replace(".", "", 1).isdigit():
        return _aspas_num(float(s))
    return f'"{v}"'


def _comparar_com_peticoes_treinadas(
    texto_usuario: str, chunks: list[dict]
) -> dict[str, Any]:
    """
    Agrupa chunks recuperados por PDF treinado e mede semelhança petição-a-petição.
    Agora considera também o tipo de documento para filtrar comparações relevantes.
    """
    por_doc: dict[str, list[float]] = defaultdict(list)
    trechos_doc: dict[str, list[str]] = defaultdict(list)
    tipos_doc: dict[str, str] = {}

    for c in chunks:
        meta = c.get("metadata") or {}
        src = _nome_documento(meta.get("source", ""))
        rel = float(c.get("relevancia", 0))
        por_doc[src].append(rel)
        trechos_doc[src].append((c.get("texto") or "")[:800])
        # Guarda o tipo de documento se disponível
        if src not in tipos_doc:
            tipos_doc[src] = meta.get("tipo_documento", "outros")

    ranking = []
    for doc, rels in por_doc.items():
        media = sum(rels) / len(rels)
        pct = round(media * 100, 1)
        nivel = "alta" if pct >= 70 else "media" if pct >= 45 else "baixa"
        ranking.append(
            {
                "documento": doc,
                "similaridade_pct": pct,
                "nivel": nivel,
                "chunks_comparados": len(rels),
                "tipo_documento": tipos_doc.get(doc, "outros"),
                "trecho_referencia": "\n".join(trechos_doc.get(doc, []))[:2500],
            }
        )
    ranking.sort(key=lambda x: x["similaridade_pct"], reverse=True)

    # Lacunas: tópicos na melhor referência ausentes na petição do usuário
    lacunas: list[dict] = []
    if ranking and ranking[0].get("trecho_referencia"):
        ref_txt = ranking[0]["trecho_referencia"]
        for chave, (titulo, pat) in TOPICOS.items():
            if pat.search(ref_txt) and not pat.search(texto_usuario):
                lacunas.append(
                    {
                        "topico": titulo,
                        "presente_na_referencia": ranking[0]["documento"],
                        "presente_na_sua": False,
                    }
                )

    return {
        "total_peticoes_comparadas": len(ranking),
        "ranking": ranking,
        "lacunas_vs_melhor_referencia": lacunas,
        "melhor_referencia": ranking[0]["documento"] if ranking else None,
    }


def _score_rag(texto: str, k: int | None = None) -> tuple[float, dict]:
    k = k or COMPARACAO_TOP_K
    query = texto[:2500].strip()
    rec = recuperar_chunks(query, k=k)
    chunks = rec.get("chunks") or []
    if not chunks:
        return 0.0, rec
    relevancias = [float(c.get("relevancia", 0)) for c in chunks]
    media = sum(relevancias) / len(relevancias)
    return min(100, media * 100), rec


def _score_completude(texto: str) -> tuple[float, list[str]]:
    notas: list[str] = []
    pts = 0.0
    if len(texto) >= 1200:
        pts += 35
    elif len(texto) >= 800:
        pts += 22
        notas.append("Texto curto — petições sólidas costumam ter mais de 1.200 caracteres.")
    else:
        notas.append("Petição muito curta — desenvolva fatos, direito e pedidos.")
    t = texto.lower()
    if re.search(r"jurisprud|súmula|sumula|resp|stj|stf", t):
        pts += 35
    else:
        notas.append("Inclua jurisprudência ou súmula aplicável na fundamentação.")
    if "pedidos" in t or "requer" in t:
        pts += 30
    else:
        notas.append("Explicitar secção de pedidos (requer, pede, dos pedidos).")
    return min(100, pts), notas


def _score_completude_llm(texto: str) -> tuple[float, list[str]] | None:
    """
    Usa o LLM para avaliar completude semântica real.
    Retorna (pontuação, observações) ou None se falhar.
    """
    try:
        from backend.llm_local import get_json_llm

        chain = _PROMPT_COMPLETUDE | get_json_llm(temperature=0.1)
        raw = chain.invoke({"peticao": texto[:4000]})
        content = raw.content if hasattr(raw, "content") else str(raw)
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```\w*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
        data = json.loads(content)
        pontuacao = float(data.get("pontuacao", 50))
        observacoes = data.get("observacoes") or []
        return min(100, pontuacao), observacoes
    except Exception:
        return None


def _normalizar_melhoria(item: Any) -> dict | None:
    if isinstance(item, str) and item.strip():
        return {
            "secao": "Geral",
            "problema": item.strip(),
            "sugestao": "Revise com base nas petições treinadas em data/knowledge/.",
            "prioridade": "media",
        }
    if not isinstance(item, dict):
        return None
    secao = (
        item.get("secao")
        or item.get("secção")
        or item.get("section")
        or item.get("documento")
        or "Geral"
    )
    problema = item.get("problema") or item.get("problem") or item.get("lacuna") or ""
    sugestao = (
        item.get("sugestao")
        or item.get("sugestão")
        or item.get("suggestion")
        or item.get("acao")
        or ""
    )
    if not str(problema).strip() and not str(sugestao).strip():
        return None
    return {
        "secao": str(secao).strip(),
        "problema": str(problema).strip() or "Ajuste necessário face aos modelos.",
        "sugestao": str(sugestao).strip()
        or "Consulte a petição de referência indicada e alinhe redação e estrutura.",
        "prioridade": (item.get("prioridade") or item.get("priority") or "media").lower(),
        "referencia_modelo": item.get("referencia_modelo") or item.get("documento") or "",
    }


def _melhorias_heuristicas(validacao: dict, completude_notas: list[str]) -> list[dict]:
    melhorias: list[dict] = []
    for t in validacao.get("topicos", []):
        if t.get("presente"):
            continue
        titulo = t.get("titulo", t.get("id", "Secção"))
        melhorias.append(
            {
                "secao": titulo,
                "problema": f"Secção «{titulo}» não identificada na sua petição.",
                "sugestao": f"Inclua título e conteúdo explícito para «{titulo}», como nas petições modelo.",
                "prioridade": "alta",
            }
        )
    for n in completude_notas:
        melhorias.append(
            {
                "secao": "Completude",
                "problema": n,
                "sugestao": "Compare com os PDFs treinados e expanda fundamentação e pedidos.",
                "prioridade": "media",
            }
        )
    return melhorias


def _melhorias_da_comparacao(comparacao: dict) -> list[dict]:
    melhorias: list[dict] = []
    for lac in comparacao.get("lacunas_vs_melhor_referencia") or []:
        ref = lac.get("presente_na_referencia", "modelo")
        topico = lac.get("topico", "Tópico")
        melhorias.append(
            {
                "secao": topico,
                "problema": f"Presente em «{ref}», mas ausente ou fraco na sua petição.",
                "sugestao": f"Replique o tratamento de «{topico}» usado em {ref}.",
                "prioridade": "alta",
                "referencia_modelo": ref,
            }
        )
    for item in comparacao.get("comparacao_llm") or []:
        if not isinstance(item, dict):
            continue
        falta = (item.get("o_que_falta_na_sua") or "").strip()
        doc = item.get("documento", "referência")
        if falta and falta.lower() not in ("nenhum", "n/a", "—", "-", "na"):
            melhorias.append(
                {
                    "secao": doc,
                    "problema": falta,
                    "sugestao": f"Ajuste a redação para aproximar de `{doc}` (semelhança: {item.get('semelhanca', '?')}).",
                    "prioridade": "alta"
                    if str(item.get("semelhanca", "")).lower() == "baixa"
                    else "media",
                    "referencia_modelo": doc,
                }
            )
    for r in comparacao.get("ranking") or []:
        pct = float(r.get("similaridade_pct", 100))
        if pct < 55:
            doc = r.get("documento", "?")
            melhorias.append(
                {
                    "secao": "Comparação",
                    "problema": f"Semelhança baixa ({pct}%) com a petição treinada «{doc}».",
                    "sugestao": f"Revise estrutura, pedidos e fundamentação usando «{doc}» como referência principal.",
                    "prioridade": "alta" if pct < 40 else "media",
                    "referencia_modelo": doc,
                }
            )
    return melhorias


def _melhorias_fallback(
    *,
    validacao: dict,
    classificacao: dict,
    comparacao: dict,
    scores: dict,
) -> list[dict]:
    """Garante pelo menos sugestões quando heurística e LLM não preencheram."""
    out: list[dict] = []
    label = classificacao.get("label", "")
    if label == "DEU RUIM":
        out.append(
            {
                "secao": "Qualidade global",
                "problema": "Classificação heurística: DEU RUIM — estrutura ou conteúdo insuficiente.",
                "sugestao": "Reestruture a petição com qualificação, fatos, direito, pedidos e valor da causa.",
                "prioridade": "alta",
            }
        )
    elif label == "MEIO CERTO":
        out.append(
            {
                "secao": "Qualidade global",
                "problema": "Classificação MEIO CERTO — petição aceitável mas abaixo dos modelos.",
                "sugestao": "Reforce jurisprudência, pedidos específicos e valor da causa.",
                "prioridade": "media",
            }
        )
    if scores.get("estrutura_topicos", 100) < 70:
        out.append(
            {
                "secao": "Estrutura",
                "problema": f"Cobertura de tópicos baixa ({scores.get('estrutura_topicos')}%).",
                "sugestao": "Use as secções das petições em data/knowledge/ como checklist.",
                "prioridade": "alta",
            }
        )
    if scores.get("comparacao_peticoes_treinadas", 100) < 50:
        ref = comparacao.get("melhor_referencia") or "petição modelo"
        out.append(
            {
                "secao": "Aderência aos modelos",
                "problema": "Baixa similaridade global com as petições treinadas.",
                "sugestao": f"Compare parágrafo a parágrafo com `{ref}`.",
                "prioridade": "alta",
                "referencia_modelo": ref,
            }
        )
    if not out and comparacao.get("melhor_referencia"):
        out.append(
            {
                "secao": "Refinamento",
                "problema": "Petição razoável; ainda há margem face ao melhor modelo.",
                "sugestao": f"Leia `{comparacao['melhor_referencia']}` e alinhe tom, pedidos e citações.",
                "prioridade": "baixa",
                "referencia_modelo": comparacao["melhor_referencia"],
            }
        )
    return out


def _consolidar_melhorias(*fontes: list) -> list[dict]:
    vistos: set[tuple[str, str]] = set()
    merged: list[dict] = []
    for fonte in fontes:
        for raw in fonte or []:
            m = _normalizar_melhoria(raw)
            if not m:
                continue
            chave = (m["secao"].lower(), m["problema"][:100].lower())
            if chave in vistos:
                continue
            vistos.add(chave)
            merged.append(m)
    return merged


def _analise_ollama(
    texto: str,
    chunks: list[dict],
    validacao: dict,
    classificacao: dict,
    comparacao: dict,
) -> dict | None:
    if not chunks:
        return None
    exemplos = "\n\n---\n\n".join(
        f"[Petição treinada: {_nome_documento((c.get('metadata') or {}).get('source', '?'))}]\n"
        f"{(c.get('texto') or '')[:1200]}"
        for c in chunks[:5]
    )
    ranking_lines = []
    for r in comparacao.get("ranking", [])[:8]:
        ranking_lines.append(
            f"- {r['documento']}: {r['similaridade_pct']}% ({r['nivel']})"
        )
    ranking_docs = "\n".join(ranking_lines) or "nenhum"
    faltando = ", ".join(t["titulo"] for t in validacao.get("faltando", [])) or "nenhum"
    try:
        # Usa get_json_llm para garantir resposta JSON válida
        from backend.llm_local import get_json_llm

        chain = _PROMPT_AVALIACAO | get_json_llm(temperature=0.1)
        raw = chain.invoke(
            {
                "exemplos": exemplos,
                "ranking_docs": ranking_docs,
                "peticao": texto[:6000],
                "faltando": faltando,
                "label": classificacao.get("label", "?"),
            }
        )
        content = raw.content if hasattr(raw, "content") else str(raw)
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```\w*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
        return json.loads(content)
    except json.JSONDecodeError:
        return _extrair_json_parcial(content)  # type: ignore[possibly-undefined]
    except Exception:
        return None


def _extrair_json_parcial(content: str) -> dict | None:
    """Tenta recuperar melhorias mesmo se o JSON do LLM vier truncado."""
    out: dict = {"melhorias": [], "pontos_fortes": [], "analise_critica": "", "resumo": ""}
    m = re.search(r'"analise_critica"\s*:\s*"((?:[^"\\]|\\.)*)"', content, re.DOTALL)
    if m:
        out["analise_critica"] = m.group(1).replace('\\"', '"')
    bloco = re.search(r'"melhorias"\s*:\s*\[(.*)\]', content, re.DOTALL)
    if bloco:
        for item_m in re.finditer(
            r'\{[^{}]*"secao"[^{}]*\}', bloco.group(1), re.DOTALL | re.IGNORECASE
        ):
            try:
                item = json.loads(item_m.group(0))
                out["melhorias"].append(item)
            except json.JSONDecodeError:
                continue
    return out if out["melhorias"] or out["analise_critica"] else None


def avaliar_peticao(
    texto: str,
    *,
    usar_ollama: bool = True,
    k_rag: int | None = None,
) -> dict[str, Any]:
    texto = (texto or "").strip()
    if len(texto) < 80:
        return {"erro": "Texto muito curto (mínimo ~80 caracteres)."}

    s_estrutura, validacao = _score_estrutura(texto)
    s_class, classificacao = _score_classificacao(texto)
    s_rag, rec = _score_rag(texto, k=k_rag or COMPARACAO_TOP_K)
    chunks = rec.get("chunks") or []
    comparacao = _comparar_com_peticoes_treinadas(texto, chunks)
    s_comp, notas_comp = _score_completude(texto)

    # Tenta score de completude via LLM (mais preciso)
    completude_llm = None
    if usar_ollama:
        completude_llm = _score_completude_llm(texto)
    if completude_llm:
        s_comp_llm, obs_llm = completude_llm
        # Pondera: 60% heurística, 40% LLM
        s_comp = round(s_comp * 0.6 + s_comp_llm * 0.4, 1)
        notas_comp.extend(obs_llm)

    pesos = {"estrutura": 0.30, "comparacao": 0.35, "classificacao": 0.20, "completude": 0.15}
    score_final = round(
        s_estrutura * pesos["estrutura"]
        + s_rag * pesos["comparacao"]
        + s_class * pesos["classificacao"]
        + s_comp * pesos["completude"],
        1,
    )

    if rec.get("aviso"):
        score_final = round(score_final * 0.5, 1)

    comparacao_llm: list[dict] = []
    melhorias_llm: list = []
    pontos_fortes: list[str] = []
    resumo_llm = ""
    analise_critica = ""
    ollama_offline = False

    analise = None
    if usar_ollama and chunks:
        analise = _analise_ollama(texto, chunks, validacao, classificacao, comparacao)
    if analise:
        raw_m = analise.get("melhorias")
        if isinstance(raw_m, list):
            melhorias_llm = raw_m
        pontos_fortes = analise.get("pontos_fortes") or []
        resumo_llm = analise.get("resumo", "")
        analise_critica = analise.get("analise_critica", "")
        comparacao_llm = analise.get("comparacao_por_referencia") or []
    elif usar_ollama and chunks:
        # Se não veio análise mesmo com ollama ativo, marca como offline
        ollama_offline = True

    comparacao_completa = {**comparacao, "comparacao_llm": comparacao_llm}
    scores_parciais = {
        "estrutura_topicos": s_estrutura,
        "comparacao_peticoes_treinadas": s_rag,
        "classificacao": s_class,
        "completude": s_comp,
    }
    melhorias = _consolidar_melhorias(
        _melhorias_heuristicas(validacao, notas_comp),
        _melhorias_da_comparacao(comparacao_completa),
        melhorias_llm,
        _melhorias_fallback(
            validacao=validacao,
            classificacao=classificacao,
            comparacao=comparacao_completa,
            scores=scores_parciais,
        ),
    )

    nivel = (
        "excelente"
        if score_final >= 80
        else "bom"
        if score_final >= 65
        else "regular"
        if score_final >= 45
        else "insuficiente"
    )

    resultado = {
        "score_final": score_final,
        "nivel": nivel,
        "scores": {
            "estrutura_topicos": round(s_estrutura, 1),
            "comparacao_peticoes_treinadas": round(s_rag, 1),
            "classificacao": round(s_class, 1),
            "completude": round(s_comp, 1),
        },
        "comparacao": comparacao_completa,
        "pesos": pesos,
        "classificacao": classificacao,
        "validacao_topicos": validacao,
        "recuperacao": {
            "chunks_usados": len(rec.get("chunks") or []),
            "fontes": [
                {
                    "source": c.get("metadata", {}).get("source"),
                    "relevancia": c.get("relevancia"),
                    "tipo_documento": c.get("metadata", {}).get("tipo_documento"),
                }
                for c in (rec.get("chunks") or [])[:5]
            ],
        },
        "pontos_fortes": pontos_fortes,
        "analise_critica": analise_critica,
        "melhorias": melhorias,
        "resumo": resumo_llm
        or (
            f"Score {score_final}/100 ({nivel}). "
            f"Comparada com {comparacao.get('total_peticoes_comparadas', 0)} petição(ões) treinada(s)."
        ),
        "aviso_base": rec.get("aviso"),
        "ollama_offline": ollama_offline,
    }

    return resultado


def formatar_relatorio_markdown(resultado: dict) -> str:
    if resultado.get("erro"):
        return f"**Erro:** {resultado['erro']}"

    ac = (resultado.get("analise_critica") or "").strip()
    sc = resultado["scores"]
    linhas = [
        f"## Score global: {_aspas_num(resultado['score_final'])} / 100 — *{resultado['nivel'].upper()}*",
        f"**Classificação:** {resultado['classificacao'].get('label', '—')}",
        "",
    ]
    if resultado.get("ollama_offline"):
        linhas.append(
            "⚠️ **Ollama offline ou indisponível.** A análise crítica com IA não foi gerada. "
            "Verifique se o serviço Ollama está em execução."
        )
        linhas.append("")
    if ac:
        linhas.extend(["### Análise crítica", ac, ""])
    linhas.extend([
        "### Pontuação por dimensão",
        f"- Estrutura (tópicos): {_aspas_num(sc['estrutura_topicos'])}",
        f"- Comparação com petições treinadas: {_aspas_num(sc.get('comparacao_peticoes_treinadas', sc.get('similaridade_base', 0)))}",
        f"- Classificação: {_aspas_num(sc['classificacao'])} ({resultado['classificacao']['label']})",
        f"- Completude: {_aspas_num(sc['completude'])}",
        "",
        f"*{resultado.get('resumo', '')}*",
    ])
    if resultado.get("aviso_base"):
        linhas.extend(["", f"⚠️ {resultado['aviso_base']}"])

    if resultado.get("pontos_fortes"):
        linhas.extend(["", "### Pontos fortes"])
        for p in resultado["pontos_fortes"]:
            linhas.append(f"- {p}")

    melhorias = resultado.get("melhorias") or []
    if melhorias:
        linhas.extend(["", "### Onde melhorar"])
        for i, m in enumerate(melhorias[:12], 1):
            if isinstance(m, dict):
                pri = m.get("prioridade", "")
                linhas.append(
                    f"{i}. **{m.get('secao', 'Geral')}** ({pri})\n"
                    f"   - Problema: {m.get('problema', '')}\n"
                    f"   - Sugestão: {m.get('sugestao', '')}"
                )
            else:
                linhas.append(f"{i}. {m}")

    comp = resultado.get("comparacao") or {}
    ranking = comp.get("ranking") or []
    if ranking:
        linhas.extend(["", "### Comparação com petições treinadas"])
        for r in ranking:
            tipo = r.get("tipo_documento", "")
            tipo_str = f" ({tipo})" if tipo and tipo != "outros" else ""
            linhas.append(
                f"- **{r['documento']}**{tipo_str} — semelhança {_aspas_num(r['similaridade_pct'])} ({r['nivel']})"
            )
    for item in comp.get("comparacao_llm") or []:
        if isinstance(item, dict):
            linhas.extend(
                [
                    "",
                    f"**{item.get('documento', '?')}** ({item.get('semelhanca', '')})",
                    f"- Na sua petição falta: {item.get('o_que_falta_na_sua', '—')}",
                    f"- O que está bem: {item.get('o_que_esta_bem', '—')}",
                ]
            )
    lac = comp.get("lacunas_vs_melhor_referencia") or []
    if lac:
        linhas.extend(["", "### Lacunas face à melhor referência"])
        for L in lac:
            linhas.append(
                f"- {L.get('topico')}: presente em `{L.get('presente_na_referencia')}`, ausente na sua"
            )

    fontes = resultado.get("recuperacao", {}).get("fontes") or []
    if fontes:
        linhas.extend(["", "### Trechos recuperados"])
        for f in fontes:
            tipo = f.get("tipo_documento", "")
            tipo_str = f" ({tipo})" if tipo and tipo != "outros" else ""
            linhas.append(
                f"- `{f.get('source')}`{tipo_str} (relevância {_aspas_num(f.get('relevancia'))})"
            )

    faltando = resultado.get("validacao_topicos", {}).get("faltando") or []
    if faltando:
        linhas.extend(["", "### Tópicos em falta (heurística)"])
        for t in faltando:
            tit = t.get("titulo", t.get("id", ""))
            linhas.append(f"- {tit}")

    return "\n".join(linhas)


def preparar_dashboard(resultado: dict) -> dict:
    """Converte resultado da avaliação em componentes do dashboard Gradio."""
    if resultado.get("erro"):
        return {"erro": resultado["erro"]}

    scores = resultado.get("scores") or {}
    dim_labels = {
        "estrutura_topicos": "Estrutura (tópicos)",
        "comparacao_peticoes_treinadas": "Comparação c/ petições treinadas",
        "classificacao": "Qualidade geral",
        "completude": "Completude",
    }
    if "comparacao_peticoes_treinadas" not in scores and "similaridade_base" in scores:
        scores["comparacao_peticoes_treinadas"] = scores["similaridade_base"]
    tabela_scores = [
        [dim_labels.get(k, k), _aspas_num(scores.get(k, 0))]
        for k in dim_labels
    ]

    topicos = []
    for t in resultado.get("validacao_topicos", {}).get("topicos", []):
        topicos.append(
            [
                t.get("titulo", ""),
                "Sim" if t.get("presente") else "Não",
                t.get("evidencia", "")[:60],
            ]
        )

    melhorias_rows = []
    for m in resultado.get("melhorias") or []:
        norm = _normalizar_melhoria(m)
        if norm:
            ref = norm.get("referencia_modelo") or ""
            sugestao = norm["sugestao"]
            if ref:
                sugestao = f"{sugestao} (ref.: {ref})"
            melhorias_rows.append(
                [
                    norm["secao"],
                    norm["prioridade"],
                    norm["problema"],
                    sugestao,
                ]
            )
    if not melhorias_rows:
        melhorias_rows = [
            [
                "—",
                "—",
                "Não foi possível gerar melhorias. Treine a base e verifique o Ollama.",
                "Execute «Treinar petições da base» e repita a análise.",
            ]
        ]

    tabela_comparacao = []
    for r in (resultado.get("comparacao") or {}).get("ranking") or []:
        tabela_comparacao.append(
            [
                r.get("documento", ""),
                _aspas_num(r.get("similaridade_pct", 0)),
            ]
        )

    tabela_lacunas = []
    for L in (resultado.get("comparacao") or {}).get("lacunas_vs_melhor_referencia") or []:
        tabela_lacunas.append(
            [
                L.get("topico", ""),
                L.get("presente_na_referencia", ""),
                "Sim" if L.get("presente_na_sua") else "Não",
            ]
        )

    comparacao_md = _formatar_comparacao_md(resultado.get("comparacao") or {})

    pontos = resultado.get("pontos_fortes") or []
    return {
        "score": resultado["score_final"],
        "nivel": resultado["nivel"].upper(),
        "classificacao": resultado["classificacao"].get("label", "—"),
        "resumo": resultado.get("resumo", ""),
        "analise_critica": resultado.get("analise_critica") or resultado.get("resumo", ""),
        "pontos_fortes": "\n".join(f"• {p}" for p in pontos) or "—",
        "tabela_scores": tabela_scores,
        "tabela_topicos": topicos,
        "tabela_melhorias": melhorias_rows,
        "tabela_comparacao": tabela_comparacao,
        "tabela_lacunas": tabela_lacunas,
        "comparacao_md": comparacao_md,
        "melhor_referencia": (resultado.get("comparacao") or {}).get("melhor_referencia")
        or "—",
        "relatorio_md": formatar_relatorio_markdown(resultado),
        "aviso": resultado.get("aviso_base") or "",
        "ollama_offline": resultado.get("ollama_offline", False),
    }


def _formatar_comparacao_md(comparacao: dict) -> str:
    if not comparacao:
        return "_Treine a base com PDFs em data/knowledge/ e analise novamente._"
    linhas = [
        "## Comparação: a sua petição × petições treinadas",
        "",
    ]
    if comparacao.get("melhor_referencia"):
        linhas.append(
            f"Petição treinada mais próxima: **{comparacao['melhor_referencia']}**"
        )
    linhas.append("")
    for r in comparacao.get("ranking") or []:
        tipo = r.get("tipo_documento", "")
        tipo_str = f" ({tipo})" if tipo and tipo != "outros" else ""
        linhas.append(
            f"- **{r['documento']}**{tipo_str} — semelhança {_aspas_num(r['similaridade_pct'])} ({r['nivel']})"
        )
    for item in comparacao.get("comparacao_llm") or []:
        if not isinstance(item, dict):
            continue
        linhas.extend(
            [
                "",
                f"### vs `{item.get('documento', '?')}` ({item.get('semelhanca', '')})",
                f"**Falta na sua petição:** {item.get('o_que_falta_na_sua', '—')}",
                f"**O que está bem:** {item.get('o_que_esta_bem', '—')}",
            ]
        )
    lac = comparacao.get("lacunas_vs_melhor_referencia") or []
    if lac:
        linhas.extend(["", "**Tópicos na referência que faltam na sua:**"])
        for L in lac:
            linhas.append(f"- {L.get('topico')} (em `{L.get('presente_na_referencia')}`)")
    return "\n".join(linhas)
