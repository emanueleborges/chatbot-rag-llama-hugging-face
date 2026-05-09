"""Mapa do chat: Nominatim (OSM) ou — com chave — Open Charge Map (pontos de recarga reais).

Google Maps / Places exige API paga; OCM é o padrão comunitário para estações VE.
Documentação: https://openchargemap.org/site/develop/api
"""
from __future__ import annotations

import os
import re
from typing import Any

import httpx

from config import openchargemap_api_key, openchargemap_max_results, openchargemap_radius_km
from geocode_service import (
    _extract_place_query,
    _wants_location_query,
    geocode_if_location_query,
    nominatim_lookup_place,
)

_OCM_BASE = "https://api.openchargemap.io/v3/poi/"

# Intenção explícita de listar / encontrar postos de recarga (activa OCM com chave).
_CHARGING_POI_INTENT = re.compile(
    r"(loca(?:is|l)\s+de\s+recarga|pontos?\s+de\s+recarga|eletroposto|postos?\s+de\s+recarga|"
    r"esta[cç](?:ão|ões)\s+de\s+recarga|infraestrutura\s+de\s+recarga|"
    r"onde\s+(?:carregar|recarregar)|carregador(?:es)?\s+público|carregadores?\s+públicos|"
    r"tomada\s+pública\s+de\s+recarga)",
    re.I | re.UNICODE,
)


def _ocm_user_agent() -> str:
    return os.environ.get("OPENCHARGEMAP_USER_AGENT", "").strip() or "RAG-Chat-EV/1.0"


def _wants_charging_stations_poi(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 6:
        return False
    return bool(_CHARGING_POI_INTENT.search(t))


def _fetch_open_charge_markers(lat: float, lng: float, api_key: str) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "output": "json",
        "latitude": lat,
        "longitude": lng,
        "distance": openchargemap_radius_km(),
        "distanceunit": "KM",
        "maxresults": openchargemap_max_results(),
        "countrycode": "BR",
        "compact": "true",
        "key": api_key,
    }
    headers = {"User-Agent": _ocm_user_agent(), "Accept": "application/json"}
    try:
        with httpx.Client(timeout=18.0, headers=headers, follow_redirects=True) as client:
            r = client.get(_OCM_BASE, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    markers: list[dict[str, Any]] = []
    for row in data:
        ai = row.get("AddressInfo") if isinstance(row, dict) else None
        if not isinstance(ai, dict):
            continue
        try:
            plat = float(ai.get("Latitude"))
            plng = float(ai.get("Longitude"))
        except (TypeError, ValueError):
            continue
        title = (ai.get("Title") or "Estação de recarga")[:140]
        town = (ai.get("Town") or "").strip()
        addr = (ai.get("AddressLine1") or "").strip()
        subtitle = ", ".join(x for x in (addr, town) if x)[:200]
        markers.append({"lat": plat, "lng": plng, "title": title, "subtitle": subtitle})
    return markers


def build_chat_map_payload(text: str) -> dict[str, Any] | None:
    """
    Devolve payload para o campo JSON `map`:
    - Com OPENCHARGEMAP_API_KEY e pergunta sobre postos: vários marcadores (OCM).
    - Caso contrário: um ponto (Nominatim) como antes.
    """
    if not _wants_location_query(text):
        return None

    key = openchargemap_api_key()
    if key and _wants_charging_stations_poi(text):
        q = _extract_place_query(text)
        center = nominatim_lookup_place(q)
        if not center:
            return geocode_if_location_query(text)

        markers = _fetch_open_charge_markers(center["lat"], center["lng"], key)
        place_short = (center.get("label") or q)[:120]
        n = len(markers)
        summary = (
            f"{n} ponto(s) de recarga (Open Charge Map) num raio de ~{openchargemap_radius_km()} km "
            f"— referência: {place_short}"
        )
        out: dict[str, Any] = {
            "lat": center["lat"],
            "lng": center["lng"],
            "zoom": 12,
            "label": summary,
            "markers": markers,
            "source": "openchargemap",
            "attribution": "Dados: Open Charge Map (colaborativo) · Mapa: © OpenStreetMap",
        }
        if n == 0:
            out["label"] = (
                f"Nenhum posto listado no Open Charge Map neste raio (~{openchargemap_radius_km()} km). "
                f"Área: {place_short}"
            )
        return out

    return geocode_if_location_query(text)
