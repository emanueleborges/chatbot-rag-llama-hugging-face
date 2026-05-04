"""Testes do roteador (sem rede nem API keys)."""
from routing import image_intent, route_message


def test_route_lenovo():
    assert route_message("Fale sobre ThinkPad") == "rag_lenovo"
    assert route_message("computadores Le Novo") == "rag_lenovo"


def test_route_dell():
    assert route_message("Dell XPS garantia") == "rag"


def test_route_llm_generic():
    assert route_message("Olá, como estás?") == "llm"


def test_image_intent():
    assert image_intent("crie uma imagem de um gato") is True
    assert image_intent("Qual é a capital de Portugal?") is False
