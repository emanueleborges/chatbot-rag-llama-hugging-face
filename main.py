"""Inicia o servidor web RAG Chat (abra http://127.0.0.1:8000 ou defina UVICORN_HOST)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import uvicorn

if __name__ == "__main__":
    host = os.environ.get("UVICORN_HOST", "127.0.0.1")
    port = int(os.environ.get("UVICORN_PORT", "8000"))
    reload = os.environ.get("UVICORN_RELOAD", "1").strip().lower() not in ("0", "false", "no")
    uvicorn.run("server:app", host=host, port=port, reload=reload)
