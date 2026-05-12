# database.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import redis
import json
import hashlib
import os

class VectorDatabase:
    def __init__(self, cache_host="localhost", cache_port=6379):
        # ChromaDB persistente
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))
        
        # Collection para documentos
        self.collection = self.client.get_or_create_collection(
            name="documentos",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Modelo de embedding local
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Redis para cache
        self.cache = redis.Redis(
            host=cache_host,
            port=cache_port,
            decode_responses=True,
            socket_timeout=5
        )
        
        # Configurações de chunking
        self.chunk_size = 1000   # caracteres
        self.chunk_overlap = 200  # sobreposição
        
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Divide texto em chunks inteligentes com sobreposição"""
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap
        
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # Busca quebra natural (espaço ou pontuação)
            if end < text_len:
                # Procura por quebra de linha
                newline = text.rfind('\n', start, end)
                if newline > start:
                    end = newline + 1
                else:
                    # Procura por espaço
                    space = text.rfind(' ', start, end)
                    if space > start:
                        end = space + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Avança com sobreposição
            start = end - overlap if end < text_len else end
        
        return chunks
    
    def add_document(self, text: str, metadata: Dict = None) -> List[str]:
        """Adiciona documento processado em chunks com cache"""
        metadata = metadata or {}
        
        # Gera ID único baseado no conteúdo
        doc_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Verifica se já existe no cache
        cached = self.cache.get(f"doc:{doc_hash}")
        if cached:
            return json.loads(cached)
        
        # Cria chunks do documento
        chunks = self.chunk_text(text)
        ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_hash}_{i}"
            embedding = self.embedder.encode(chunk).tolist()
            
            self.collection.add(
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    **metadata,
                    "doc_id": doc_hash,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }],
                ids=[chunk_id]
            )
            ids.append(chunk_id)
        
        # Armazena no cache
        self.cache.setex(f"doc:{doc_hash}", 3600, json.dumps(ids))
        
        return ids
    
    def search(self, query: str, k: int = 5, use_cache: bool = True) -> List[Dict]:
        """Busca chunks relevantes com cache"""
        # Verifica cache da query
        cache_key = f"query:{hashlib.md5(query.encode()).hexdigest()}"
        
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                return json.loads(cached)
        
        # Gera embedding da query
        query_embedding = self.embedder.encode(query).tolist()
        
        # Busca no ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Formata resultados
        documents = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            documents.append({
                "text": doc,
                "metadata": metadata,
                "relevance": 1 - distance
            })
        
        # Armazena em cache por 5 minutos
        self.cache.setex(cache_key, 300, json.dumps(documents))
        
        return documents
    
    def clear_cache(self):
        """Limpa o cache Redis"""
        self.cache.flushdb()