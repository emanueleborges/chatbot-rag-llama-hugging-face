"""Smoke test: uma volta ao LLM via chat_engine (usa OLLAMA_* do ambiente)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


async def main() -> int:
    from chat_engine import process_chat

    r = await process_chat("Responda apenas com a palavra: OK", [])
    if r.get("error"):
        print("ERRO:", r.get("content"), file=sys.stderr)
        return 1
    text = (r.get("content") or "").strip()
    if not text:
        print("Resposta vazia do LLM.", file=sys.stderr)
        return 1
    print("OK — resposta:", text[:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
