"""Classificação de petições: DEU RUIM / DEU BOM / MEIO CERTO."""
from __future__ import annotations

import re
from pathlib import Path

from backend.config import LABELS, MODELS_DIR

_TOPICOS = {
    "qualificacao": re.compile(
        r"\b(autor|réu|requerente|requerido|cpf|cnpj|qualifica)\b", re.I
    ),
    "fatos": re.compile(r"\b(dos fatos|fatos|historico|narra)\b", re.I),
    "fundamentacao": re.compile(
        r"\b(fundament|direito|art\.?\s*\d|artigo\s+\d|cdc|cf/88)\b", re.I
    ),
    "pedidos": re.compile(r"\b(pede|requer|postula|dos pedidos|pedidos)\b", re.I),
    "valor_causa": re.compile(r"\b(valor da causa|R\$\s*[\d\.]+)\b", re.I),
    "jurisprudencia": re.compile(
        r"\b(jurisprud|súmula|sumula|precedente|resp|stj|stf|acórdão)\b", re.I
    ),
}


def _cobertura_topicos(texto: str) -> tuple[float, int]:
    presentes = sum(1 for pat in _TOPICOS.values() if pat.search(texto))
    total = len(_TOPICOS)
    return (presentes / total if total else 0.0, total - presentes)


def classificar_heuristica(texto: str) -> dict:
    cob, n_faltas = _cobertura_topicos(texto)
    t = texto.lower()
    tem_jur = bool(re.search(r"jurisprud|súmula|sumula|resp|stj", t))
    tem_pedido = "pedidos" in t or "requer" in t
    if cob >= 0.83 and tem_jur and tem_pedido and len(texto) > 900:
        label = "DEU BOM"
    elif cob < 0.5 or n_faltas >= 4 or len(texto) < 800:
        label = "DEU RUIM"
    else:
        label = "MEIO CERTO"
    return {
        "label": label,
        "probabilities": {l: (1.0 if l == label else 0.0) for l in LABELS},
        "modo": "heuristica",
        "cobertura_topicos": round(cob, 2),
    }


def classificar(texto: str, model_dir: Path | None = None) -> dict:
    model_dir = model_dir or MODELS_DIR
    cfg = model_dir / "config.json"
    if not cfg.is_file():
        return classificar_heuristica(texto)
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        id2label = {0: "DEU RUIM", 1: "DEU BOM", 2: "MEIO CERTO"}
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()
        enc = tokenizer(
            texto[:8000],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = model(**enc).logits[0]
        probs = torch.softmax(logits, dim=0).tolist()
        idx = int(torch.argmax(logits).item())
        label = model.config.id2label.get(str(idx), id2label.get(idx, LABELS[idx]))
        return {
            "label": label,
            "probabilities": {
                model.config.id2label.get(str(i), LABELS[i]): round(probs[i], 4)
                for i in range(len(probs))
            },
            "modo": "bert_finetuned",
        }
    except Exception:
        return classificar_heuristica(texto)
