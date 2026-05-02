"""Inicia o servidor web RAG Chat (abra http://127.0.0.1:8000)."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
