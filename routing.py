"""Roteamento automático: imagem, RAG Dell/Lenovo/Manaus ou LLM geral."""
import re

_IMAGE_STRICT = re.compile(
    r"(crie|gere|gerar)\s+(?:uma\s+)?(?:imagem|foto|desenho)|desenhe\s|"
    r"gerar\s+imagem|faça\s+(?:uma\s+)?(?:foto|imagem)",
    re.IGNORECASE | re.UNICODE,
)


def image_intent(text: str) -> bool:
    """Detecta pedido de geração de imagem em PT-BR (variações comuns)."""
    if not (text or "").strip():
        return False
    if _IMAGE_STRICT.search(text):
        return True
    low = text.lower()
    needles = (
        "crie uma imagem",
        "crie imagem",
        "criar uma imagem",
        "criar imagem",
        "crie uma foto",
        "criar uma foto",
        "gere uma imagem",
        "gerar uma imagem",
        "gere uma foto",
        "gerar uma foto",
        "faça uma imagem",
        "fazer uma imagem",
        "faça uma foto",
        "fazer uma foto",
        "quero uma imagem",
        "quero uma foto",
        "preciso de uma imagem",
        "preciso de uma foto",
        "gostaria de uma imagem",
        "gostaria de uma foto",
        "me mostre uma imagem",
        "mostre uma imagem",
        "geração de imagem",
        "geracao de imagem",
        "text-to-image",
        "txt2img",
        "imagem conforme",
        "foto conforme",
    )
    if any(n in low for n in needles):
        return True
    if re.search(r"quero\s+que\s+(?:você|voce)\s+(?:crie|criar|gere|gerar)\s+(?:uma\s+)?(?:imagem|foto)", low):
        return True
    if re.search(
        r"(?:pode|podia|consegue)\s+(?:criar|crie|gerar|gere)\s+(?:uma\s+)?(?:imagem|foto|ilustra)",
        low,
    ):
        return True
    # verbo + visual na mesma frase (ex.: "mostre ... imagem de um gato")
    if re.search(
        r"\b(?:crie|criar|gere|gerar|faça|fazer|desenhe|desenhar|ilustrar|mostrar)\b.{0,120}\b(?:imagem|foto|ilustra|desenho)\b",
        low,
    ):
        return True
    if re.search(
        r"\b(?:imagem|foto|ilustra(?:ção|cao))\b.{0,60}\b(?:com|de|mostrando|representando)\b",
        low,
    ) and re.search(r"\b(?:crie|criar|gere|gerar|faça|quero|preciso)\b", low):
        return True
    return False


_MANAUS_MARKERS = ("manaus", "manauara", "zona franca de manaus")
_MANAUS_LOCAL_TOPICS = (
    # comércio
    "shopping",
    "shoppings",
    "mall",
    "centro comercial",
    "centros comerciais",
    "praça de alimentação",
    "praca de alimentacao",
    # praças e cidade
    "praça",
    "praças",
    "praca",
    "pracas",
    "largo",
    "largos",
    "praça pública",
    "praca publica",
    "pracas publicas",
    "centro histórico",
    "centro historico",
    "turismo",
    "passeio",
    "pontos turísticos",
    "pontos turisticos",
    "o que visitar",
    "lugares para visitar",
    # geografia / distâncias (evita LLM a inventar km)
    "próximo",
    "próxima",
    "proximo",
    "proxima",
    "perto",
    "perto de",
    "vizinha",
    "vizinho",
    "vizinhas",
    "vizinhos",
    "distância",
    "distancia",
    "quilômetro",
    "quilometro",
    " km",
    "cidade mais",
    "cidades mais",
    "município",
    "municipio",
    "municípios",
    "municipios",
    "região metropolitana",
    "regiao metropolitana",
    "metropolitana",
)


def manaus_local_intent(text: str) -> bool:
    """Perguntas sobre locais em Manaus: shoppings, praças, centro, turismo (AM)."""
    low = (text or "").lower()
    if not any(m in low for m in _MANAUS_MARKERS):
        return False
    return any(s in low for s in _MANAUS_LOCAL_TOPICS)


_DELL = (
    "dell",
    "latitude",
    "inspiron",
    "xps",
    "alienware",
    "poweredge",
    "precision",
    "ultrasharp",
    "monitor dell",
    "notebook dell",
    "servidor dell",
    "garantia dell",
    "service tag",
    "support dell",
    "chromebook dell",
    "vostro",
    "wyse",
    "emc",
    "dell technologies",
)

_LENOVO = (
    "lenovo",
    "le novo",
    "leno vo",
    "thinkpad",
    "think book",
    "thinkbook",
    "ideapad",
    "legion",
    "yoga lenovo",
    "lenovo yoga",
    "thinkcentre",
    "think center",
    "thinkstation",
    "think station",
    "thinkvision",
    "think vision",
    "notebook lenovo",
    "pc lenovo",
    "computador lenovo",
    "computadores lenovo",
    "lenovo brasil",
    "suporte lenovo",
    "lenovo vantage",
)


def route_message(text: str) -> str:
    """Retorna 'image' | 'rag_manaus' | 'rag_lenovo' | 'rag' | 'llm'."""
    t = (text or "").strip()
    if not t:
        return "llm"
    if image_intent(t):
        return "image"
    if manaus_local_intent(t):
        return "rag_manaus"
    low = t.lower()
    if any(k in low for k in _LENOVO):
        return "rag_lenovo"
    if any(k in low for k in _DELL):
        return "rag"
    return "llm"


def extract_image_prompt(text: str) -> str:
    """Remove frases de comando e devolve o texto útil para o gerador de imagem."""
    t = (text or "").strip()
    if not t:
        return "paisagem abstrata colorida"

    steps = [
        r"^.*?quero\s+que\s+(?:você|voce)\s+(?:crie|criar|gere|gerar)\s+(?:uma\s+)?(?:imagem|foto|desenho)\s*(?:de|com|mostrando)?\s*",
        r"^(?:crie|gere|gerar|faça|fazer|desenhe)\s+(?:pra\s+mim\s+)?(?:uma\s+)?(?:imagem|foto|desenho)\s*(?:de|com|:)?\s*",
        r"^desenhe\s*(?:uma\s+)?(?:imagem\s*)?(?:de|:)?\s*",
        r"^(?:por\s+favor[, ]*)?",
        r"^.*?(?:por favor[, ]*)?(?:gere|gerar|crie|criar)\s+(?:uma\s+)?(?:imagem|foto)\s*(?:de|com)?\s*",
    ]
    for _ in range(4):
        orig = t
        for pat in steps:
            t = re.sub(pat, "", t, flags=re.IGNORECASE | re.UNICODE).strip()
        if t == orig:
            break

    return (t or text).strip()
