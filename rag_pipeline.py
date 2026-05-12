# rag_pipeline.py
import httpx
import json
import asyncio
from typing import AsyncGenerator, List, Dict, Optional
from database import VectorDatabase

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def generate_stream(self, model: str, prompt: str, system_prompt: str = None) -> AsyncGenerator[str, None]:
        """Gera resposta com streaming via Ollama"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        async with self.client.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json=payload
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
    
    async def generate_embedding(self, text: str, model: str = "nomic-embed-text") -> List[float]:
        """Gera embedding via Ollama"""
        response = await self.client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": model, "prompt": text}
        )
        data = response.json()
        return data.get("embedding", [])

class RAGPipeline:
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.ollama = OllamaClient()
        self.llm_model = "llama3.2:3b"  # Modelo local via Ollama
        self.system_prompt = """Você é um assistente útil que responde perguntas baseado APENAS no contexto fornecido.
        Se a resposta não estiver no contexto, diga honestamente que não encontrou a informação.
        Use linguagem clara e objetiva.
        """
    
    async def query_stream(
        self, 
        question: str, 
        user_id: str = "anonymous",
        use_cache: bool = True
    ) -> AsyncGenerator[str, None]:
        """Pipeline RAG completo com streaming"""
        
        # 1. Retrieve - busca chunks relevantes
        chunks = self.vector_db.search(question, k=5, use_cache=use_cache)
        
        # 2. Constroi contexto
        context = "\n\n---\n\n".join([chunk["text"] for chunk in chunks])
        
        # 3. Constroi prompt
        prompt = f"""
Contexto:
{context}

Pergunta: {question}

Responda baseado SOMENTE no contexto acima.
"""
        
        # 4. Geração com streaming
        async for token in self.ollama.generate_stream(
            model=self.llm_model,
            prompt=prompt,
            system_prompt=self.system_prompt
        ):
            yield token
    
    async def query(self, question: str, user_id: str = "anonymous") -> Dict:
        """Versão síncrona (sem streaming)"""
        chunks = self.vector_db.search(question, k=5)
        
        context = "\n\n---\n\n".join([chunk["text"] for chunk in chunks])
        
        prompt = f"""
Contexto:
{context}

Pergunta: {question}

Responda baseado SOMENTE no contexto acima.
"""
        
        # Coleta todos os tokens
        full_response = ""
        async for token in self.ollama.generate_stream(self.llm_model, prompt, self.system_prompt):
            full_response += token
        
        return {
            "answer": full_response,
            "sources": [
                {
                    "text": chunk["text"][:200],
                    "relevance": chunk["relevance"]
                } for chunk in chunks
            ]
        }