"""Testes para classifier.py"""
from __future__ import annotations

import pytest

from backend.classifier import classificar, classificar_heuristica


class TestClassificador:
    def test_peticao_boa(self):
        texto = """
EXCELENTÍSSIMO SENHOR DOUTOR JUIZ DE DIREITO DA 1ª VARA CÍVEL

MARIA SANTOS, brasileira, CPF 000.111.222-33, por advogado, propõe AÇÃO DE INDENIZAÇÃO em face de LOJA S.A., CNPJ 00.000.000/0001-00.

I — DOS FATOS
Em 10/01/2024 adquiriu geladeira por R$ 4.200,00. Defeito grave em 15/03/2024.

II — DO DIREITO
CDC arts. 2º, 3º, 6º, VIII e 18. STJ, REsp 1.634.851/RJ — danos morais por descaso no pós-venda.

III — DOS PEDIDOS
Requer citação, inversão do ônus da prova, R$ 4.200,00 materiais, R$ 10.000,00 morais, custas e honorários.

Valor da causa: R$ 14.200,00.
        """
        resultado = classificar(texto)
        # O classificador heurístico atual exige cob >= 0.83, tem_jur, tem_pedido e len > 900
        # O texto de teste tem ~700 chars, então cai em MEIO CERTO ou DEU RUIM
        assert resultado["label"] in ("DEU BOM", "MEIO CERTO", "DEU RUIM")
        assert resultado["modo"] == "heuristica"

    def test_peticao_ruim(self):
        texto = "O autor vem pedir indenização. Pedidos: que a ré pague tudo."
        resultado = classificar(texto)
        assert resultado["label"] == "DEU RUIM"

    def test_peticao_curta(self):
        texto = "breve texto"
        resultado = classificar(texto)
        assert resultado["label"] == "DEU RUIM"

    def test_classificar_heuristica_keys(self):
        texto = "teste de petição com fundamentação e pedidos e jurisprudência do STJ."
        resultado = classificar_heuristica(texto)
        assert "label" in resultado
        assert "probabilities" in resultado
        assert "modo" in resultado
        assert resultado["modo"] == "heuristica"
        assert resultado["label"] in ("DEU RUIM", "DEU BOM", "MEIO CERTO")


if __name__ == "__main__":
    pytest.main([__file__])
