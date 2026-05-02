"""Geração de imagem: Hugging Face Inference API, com fallback Pollinations (sem token)."""
from __future__ import annotations

import asyncio
import base64
import os
import random
from urllib.parse import quote

import httpx

from image_prompt_enhance import prepare_image_prompts

_DEFAULT_HF_MODEL = "runwayml/stable-diffusion-v1-5"
_HF_URL = "https://api-inference.huggingface.co/models/{model}"
_POLLINATIONS = "https://image.pollinations.ai/prompt/{q}"


async def _hf_generate(
    prompt: str,
    negative: str,
    token: str,
    model: str,
) -> tuple[str, str]:
    url = _HF_URL.format(model=model)
    payload: dict = {"inputs": prompt}
    if (negative or "").strip():
        payload["parameters"] = {"negative_prompt": negative.strip()[:800]}

    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
        if r.status_code == 503:
            raise RuntimeError("Modelo HF carregando (503). Tente de novo em alguns segundos.")
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        raw = r.content
        if "application/json" in ctype or (raw[:1] in (b"{", b"[")):
            try:
                data = r.json()
            except Exception as exc:
                raise RuntimeError(f"Resposta JSON inválida da HF: {r.text[:200]}") from exc
            if isinstance(data, list) and data and isinstance(data[0], dict):
                inner = data[0]
                if "image" in inner:
                    raw = base64.b64decode(inner["image"])
                elif "blob" in inner:
                    raw = base64.b64decode(inner["blob"])
                else:
                    raise RuntimeError("Formato JSON HF não suportado.")
            elif isinstance(data, dict) and "image" in data:
                raw = base64.b64decode(data["image"])
            elif isinstance(data, dict) and data.get("error"):
                raise RuntimeError(str(data["error"]))
            else:
                raise RuntimeError("Resposta HF em JSON sem imagem reconhecida.")
    mime = "image/jpeg" if raw[:2] == b"\xff\xd8" else "image/png"
    b64 = base64.b64encode(raw).decode("ascii")
    return mime, b64


_POLLINATIONS_UA = "Mozilla/5.0 (compatible; Dell-RAG-Chat/1.0; +local-assistant)"
_POLLINATIONS_ATTEMPTS = 6


async def _pollinations_generate(prompt: str, negative: str = "") -> tuple[str, str]:
    """
    Fallback público (sem API key). O serviço costuma responder 429 sob carga;
    retentativas com backoff e mensagem clara se esgotar.
    """
    q = quote((prompt or "abstract").strip()[:2000])
    url_base = _POLLINATIONS.format(q=q)
    neg_q = quote((negative or "").strip()[:1200]) if (negative or "").strip() else ""
    headers = {"User-Agent": _POLLINATIONS_UA}

    backoff = 4.0
    async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
        for attempt in range(_POLLINATIONS_ATTEMPTS):
            url = url_base
            if neg_q:
                url = f"{url}?negative={neg_q}"
            if attempt > 0:
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}seed={random.randint(1, 10**9)}"

            r = await client.get(url, headers=headers)

            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                try:
                    wait = float(ra) if ra is not None else min(120.0, backoff)
                except ValueError:
                    wait = min(120.0, backoff)
                await asyncio.sleep(wait)
                backoff = min(120.0, backoff * 1.75)
                continue

            if r.status_code >= 400:
                r.raise_for_status()

            raw = r.content
            if len(raw) >= 100:
                mime = "image/jpeg" if raw[:2] == b"\xff\xd8" else "image/png"
                b64 = base64.b64encode(raw).decode("ascii")
                return mime, b64

            await asyncio.sleep(min(120.0, backoff))
            backoff = min(120.0, backoff * 1.5)

    raise RuntimeError(
        "O serviço gratuito de imagens (Pollinations) está limitando pedidos (HTTP 429). "
        "Aguarde alguns minutos e tente de novo, ou configure HF_API_TOKEN para usar a "
        "Hugging Face Inference API (mais estável). Veja: "
        "https://huggingface.co/settings/tokens"
    )


async def generate_image_b64(prompt: str) -> tuple[str, str]:
    """
    Gera imagem. Ordem: HF se HF_API_TOKEN existir; se falhar ou não houver token, Pollinations.
    Defina IMAGE_BACKEND=hf|pollinations|auto (auto = padrão).
    """
    raw = (prompt or "").strip() or "cena detalhada, alta qualidade"
    pos, neg = prepare_image_prompts(raw)

    backend = (os.environ.get("IMAGE_BACKEND") or "auto").strip().lower()
    token = (os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    model = (os.environ.get("HF_IMAGE_MODEL") or _DEFAULT_HF_MODEL).strip()

    if backend == "pollinations":
        return await _pollinations_generate(pos, neg)

    if backend == "hf":
        if not token:
            raise RuntimeError(
                "IMAGE_BACKEND=hf exige HF_API_TOKEN (https://huggingface.co/settings/tokens)."
            )
        return await _hf_generate(pos, neg, token, model)

    # auto
    if token:
        try:
            return await _hf_generate(pos, neg, token, model)
        except Exception:
            pass
    return await _pollinations_generate(pos, neg)
