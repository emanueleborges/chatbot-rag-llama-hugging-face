"""Geocodificação para o mini-mapa do chat (OpenStreetMap Nominatim).

Política de uso: https://operations.osmfoundation.org/policies/nominatim/
— User-Agent identificável, pedidos moderados (uma tentativa por mensagem).
"""
from __future__ import annotations

import os
import re
from typing import Any

import httpx

_NOMINATIM = "https://nominatim.openstreetmap.org/search"

# Pedidos de mapa / localização OU perguntas sobre infraestrutura de recarga (VE).
_LOCATION_HINT = re.compile(
    r"(mapa|localiza[cç][aã]o|onde\s+fica|onde\s+est[áa]|onde\s+há|perto\s+de|próximo\s+a|proximo\s+a|"
    r"eletroposto|postos?\s+de\s+recarga|carregador(?:es)?|"
    r"loca(?:is|l)\s+de\s+recarga|local\s+de\s+recarga|pontos?\s+de\s+recarga|"
    r"esta[cç](?:ão|ões)\s+de\s+recarga|recarga\s+rápida|recarga\s+rapida|"
    r"infraestrutura\s+de\s+recarga|"
    r"endere[cç]o(\s+de|\s+do|\s+da)?|coordenadas|cidade\s+de|"
    r"estado\s+do|capital\s+do|regi[aã]o\s+de)",
    re.I | re.UNICODE,
)


def _extract_place_query(text: str) -> str:
    """
    Reduz a frase do utilizador a um termo geográfico (cidade/região) para o Nominatim.
    Ex.: «locais de recarga em Curitiba» → «Curitiba, Brasil».
    """
    t = " ".join(text.split())
    patterns = (
        r"\b(?:perto\s+de|próximo\s+a|proximo\s+a)\s+(.+?)(?:\s*[?!.]|$)",
        r"\b(?:em|na|no)\s+(.+?)(?:\s*[?!.]|$)",
        r"(?:cidade|região|regiao)\s+(?:de|do|da)\s+(.+?)(?:\s*[?!.]|$)",
    )
    for pat in patterns:
        m = re.search(pat, t, re.I | re.UNICODE)
        if not m:
            continue
        place = m.group(1).strip()
        # Corta complementos frequentes após o topónimo
        place = re.split(
            r"\s+(?:para|com|que|onde|quando|se|e)\s+",
            place,
            maxsplit=1,
        )[0].strip()
        if len(place) >= 3 and len(place) <= 100:
            if "brasil" not in place.lower():
                return f"{place}, Brasil"
            return place
    if len(t) <= 220:
        return t
    return t[:220]


def _user_agent() -> str:
    return (
        os.environ.get("NOMINATIM_USER_AGENT", "").strip()
        or "RAG-Chat-EV/1.0 (educational; local instance)"
    )


def _wants_location_query(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 4 or len(t) > 420:
        return False
    return bool(_LOCATION_HINT.search(t))


def nominatim_lookup_place(q: str) -> dict[str, Any] | None:
    """
    Geocodifica uma expressão livre para o Brasil (sem filtro de «hints» da mensagem).
    Usado pelo Open Charge Map para obter o centro da pesquisa.
    """
    q = " ".join((q or "").split())
    if len(q) < 2:
        return None
    params = {
        "q": q,
        "format": "json",
        "limit": 1,
        "countrycodes": "br",
        "accept-language": "pt-BR,pt",
    }
    headers = {"User-Agent": _user_agent(), "Accept-Language": "pt-BR,pt"}
    try:
        with httpx.Client(timeout=12.0, headers=headers, follow_redirects=True) as client:
            r = client.get(_NOMINATIM, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception:
        return None
    if not data or not isinstance(data, list):
        return None
    row = data[0]
    try:
        lat = float(row["lat"])
        lon = float(row["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    label = (row.get("display_name") or q)[:240]
    typ = (row.get("type") or "").lower()
    zoom = 11 if typ in ("administrative", "state", "region") else 13
    return {
        "lat": lat,
        "lng": lon,
        "zoom": zoom,
        "label": label,
        "source": "nominatim",
    }


def geocode_if_location_query(text: str) -> dict[str, Any] | None:
    """
    Devolve {lat, lng, zoom, label, source} ou None se não aplicável / falha.
    Limita-se ao Brasil (countrycodes=br).
    """
    if not _wants_location_query(text):
        return None
    q = _extract_place_query(text)
    return nominatim_lookup_place(q)
