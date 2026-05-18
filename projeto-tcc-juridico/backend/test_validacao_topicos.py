"""Testes para validacao_topicos.py"""
from __future__ import annotations

import pytest

from backend.validacao_topicos import TOPICOS, validar_topicos


class TestValidarTopicos:
    def test_peticao_completa(self):
        texto = """
EXCELENTÍSSIMO SENHOR DOUTOR JUIZ DE DIREITO

JOÃO SILVA, brasileiro, CPF 000.111.222-33, por advogado, propõe AÇÃO DE INDENIZAÇÃO.

I — DOS FATOS
Em 10/01/2024 adquiriu produto defeituoso.

II — DO DIREITO
CDC arts. 2º, 3º, 6º, VIII e 18. STJ, REsp 1.634.851/RJ.

III — DOS PEDIDOS
Requer citação, inversão do ônus da prova, indenização.

Valor da causa: R$ 14.200,00.
        """
        resultado = validar_topicos(texto)
        assert resultado["cobertura"] >= 0.8  # pelo menos 5 de 6 tópicos
        assert resultado["presentes"] >= 5
        assert resultado["total"] == 6

    def test_peticao_vazia(self):
        resultado = validar_topicos("")
        assert resultado["cobertura"] == 0.0
        assert resultado["presentes"] == 0
        assert len(resultado["faltando"]) == 6

    def test_peticao_minima(self):
        texto = "pede indenização. R$ 1.000,00."
        resultado = validar_topicos(texto)
        assert resultado["presentes"] >= 2  # pedidos + valor_causa

    def test_topicos_keys_match(self):
        """Verifica se as chaves do TOPICOS estão consistentes."""
        for chave, (titulo, _pat) in TOPICOS.items():
            assert isinstance(chave, str)
            assert isinstance(titulo, str)
            assert len(titulo) > 0

    def test_peticao_sem_jurisprudencia(self):
        texto = """
Qualificação das partes.
Dos fatos.
Fundamentação jurídica.
Pedidos.
Valor da causa: R$ 5.000,00.
        """
        resultado = validar_topicos(texto)
        # "Dos fatos" não é detectado pelo regex atual (precisa de "dos fatos" literal)
        # "Fundamentação" é detectado, "Pedidos" é detectado, "Valor da causa" é detectado
        # "Qualificação" é detectado via "qualifica"
        assert resultado["presentes"] >= 3  # qualificação, fundamentação, pedidos, valor
        faltando_ids = [t["id"] for t in resultado["faltando"]]
        assert "jurisprudencia" in faltando_ids


if __name__ == "__main__":
    pytest.main([__file__])
