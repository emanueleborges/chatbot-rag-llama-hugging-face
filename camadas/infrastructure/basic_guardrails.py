from __future__ import annotations

import re
import unicodedata

from camadas.domain.guardrails import GuardrailVerdict, InputSafetyPolicy, OutputSafetyPolicy


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def _redact_sensitive_patterns(text: str) -> str:
    """Redação superficial de PII comum na saída (e-mail, CPF formatado, telefone BR, cartão 16 dígitos)."""
    redacted = text
    redacted = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[e-mail]",
        redacted,
    )
    redacted = re.sub(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", "[CPF]", redacted)
    redacted = re.sub(
        r"\b(?:\+?55\s?)?\(?\d{2}\)?\s?9?\d{4}-?\d{4}\b",
        "[telefone]",
        redacted,
    )
    redacted = re.sub(
        r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b",
        "[cartão]",
        redacted,
    )
    return redacted


class BasicGuardrails(InputSafetyPolicy, OutputSafetyPolicy):
    """
    Guardrails: tamanho, injeção comum, termos de risco e redação superficial de PII na saída.
    Complemente com políticas no provedor (Gemini/OpenAI) e revisão humana quando necessário.
    """

    def __init__(
        self,
        *,
        max_input_chars: int = 8000,
        max_output_chars: int = 12000,
    ) -> None:
        self._max_in = max_input_chars
        self._max_out = max_output_chars
        self._deny_input = (
            "ignore todas as instruções",
            "ignore previous instructions",
            "system override",
            "you are now",
            "reveal your prompt",
            "mostre o prompt",
            "ignore as regras",
        )
        self._deny_output = (
            "senha do administrador",
            "admin password",
            "chave privada",
            "private key",
            "-----BEGIN",
        )
        self._injection_patterns = (
            re.compile(r"<\s*script", re.IGNORECASE),
            re.compile(r"\{\{\s*.*\}\}", re.DOTALL),
            re.compile(r"\[\s*INST\s*\]", re.IGNORECASE),
        )

    def evaluate_input(self, text: str) -> GuardrailVerdict:
        if not text or not text.strip():
            return GuardrailVerdict(False, reason="Mensagem vazia.")
        cleaned = _normalize(text)
        if len(cleaned) > self._max_in:
            return GuardrailVerdict(False, reason="Mensagem excede o tamanho permitido.")
        lower = cleaned.lower()
        for phrase in self._deny_input:
            if phrase in lower:
                return GuardrailVerdict(
                    False,
                    reason="Conteúdo de entrada não permitido.",
                )
        for rx in self._injection_patterns:
            if rx.search(cleaned):
                return GuardrailVerdict(
                    False,
                    reason="Padrão suspeito na entrada.",
                )
        return GuardrailVerdict(True, safe_text=cleaned)

    def evaluate_output(self, text: str) -> GuardrailVerdict:
        text = _redact_sensitive_patterns(text)
        if len(text) > self._max_out:
            return GuardrailVerdict(
                False,
                reason="Resposta muito longa.",
                safe_text=text[: self._max_out],
            )
        lower = text.lower()
        for phrase in self._deny_output:
            if phrase in lower:
                return GuardrailVerdict(
                    False,
                    reason="Resposta filtrada por política de segurança.",
                    safe_text=(
                        "Não posso reproduzir esse tipo de conteúdo. "
                        "Reformule a pergunta de forma mais segura."
                    ),
                )
        return GuardrailVerdict(True, safe_text=text)
