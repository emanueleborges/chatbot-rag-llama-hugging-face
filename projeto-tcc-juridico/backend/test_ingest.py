"""Testes para ingest.py"""
from __future__ import annotations

import pytest

from backend.ingest import (
    _chunk_size_adaptativo,
    _classificar_tipo_documento,
    extract_bytes,
    text_to_documents,
)


class TestClassificarTipoDocumento:
    def test_peticao_inicial(self):
        texto = "EXCELENTÍSSIMO SENHOR DOUTOR JUIZ, o autor requer indenização pelos pedidos formulados."
        assert _classificar_tipo_documento(texto) == "peticao_inicial"

    def test_sentenca(self):
        texto = "ISTO POSTO, julgo procedente o pedido para condenar o réu."
        assert _classificar_tipo_documento(texto) == "sentenca"

    def test_acordao(self):
        texto = "ACÓRDÃO: Vistos, relatados e discutidos estes autos de apelação cível."
        assert _classificar_tipo_documento(texto) == "acordao"

    def test_decisao(self):
        texto = "DESPACHO: Defiro o pedido de gratuidade de justiça."
        assert _classificar_tipo_documento(texto) == "decisao"

    def test_outros(self):
        texto = "Este é um texto genérico sem termos jurídicos específicos."
        assert _classificar_tipo_documento(texto) == "outros"


class TestChunkSizeAdaptativo:
    def test_texto_curto(self):
        texto = "a" * 1000
        assert _chunk_size_adaptativo(texto) == 1200  # CHUNK_SIZE padrão

    def test_texto_medio(self):
        texto = "a" * 30000
        size = _chunk_size_adaptativo(texto)
        assert size >= 1200
        assert size <= 1500

    def test_texto_longo(self):
        texto = "a" * 60000
        size = _chunk_size_adaptativo(texto)
        assert size >= 1200
        assert size <= 1800


class TestExtractBytes:
    def test_txt_utf8(self):
        content = "Petição jurídica de teste.".encode("utf-8")
        assert extract_bytes("teste.txt", content) == "Petição jurídica de teste."

    def test_txt_latin1(self):
        content = "Petição jurídica de teste.".encode("latin-1")
        assert extract_bytes("teste.txt", content) == "Petição jurídica de teste."

    def test_arquivo_vazio(self):
        with pytest.raises(ValueError, match="Ficheiro vazio"):
            extract_bytes("teste.txt", b"")

    def test_extensao_invalida(self):
        with pytest.raises(ValueError, match="Extensão não suportada"):
            extract_bytes("teste.exe", b"conteudo")


class TestTextToDocuments:
    def test_texto_simples(self):
        docs = text_to_documents("Texto de teste para chunking.", source="teste.txt")
        assert len(docs) >= 1
        assert docs[0].metadata["source"] == "teste.txt"
        assert docs[0].metadata["tipo_documento"] == "outros"
        assert "chunk_index" in docs[0].metadata

    def test_texto_longo_varios_chunks(self):
        texto = "Parágrafo um.\n\n" * 100
        docs = text_to_documents(texto, source="longo.txt")
        assert len(docs) > 1
        assert docs[0].metadata["total_chunks"] == len(docs)

    def test_metadata_preservada(self):
        docs = text_to_documents(
            "Texto de teste.",
            source="teste.txt",
            extra_meta={"cliente": "João", "tipo": "exemplo"},
        )
        assert docs[0].metadata["cliente"] == "João"
        assert docs[0].metadata["tipo"] == "exemplo"


if __name__ == "__main__":
    pytest.main([__file__])
