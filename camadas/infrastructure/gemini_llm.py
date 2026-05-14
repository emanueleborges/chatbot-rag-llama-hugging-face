from __future__ import annotations

import os
from collections.abc import AsyncIterator

import google.generativeai as genai


class GeminiChatLanguageModel:
    """Adaptador Gemini para a porta `ChatLanguageModel` (temperatura e modelo via env)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Defina GEMINI_API_KEY ou GOOGLE_API_KEY para usar o LLM Gemini.",
            )
        genai.configure(api_key=key)
        self._model_name = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self._temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
        )

    def _build_model(self, *, system_instruction: str) -> genai.GenerativeModel:
        return genai.GenerativeModel(
            self._model_name,
            system_instruction=system_instruction,
            generation_config=genai.GenerationConfig(temperature=self._temperature),
        )

    async def astream_completion(
        self, *, system: str, user_prompt: str
    ) -> AsyncIterator[str]:
        model = self._build_model(system_instruction=system)
        response = await model.generate_content_async(user_prompt, stream=True)
        async for chunk in response:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    async def acomplete(self, *, system: str, user_prompt: str) -> str:
        model = self._build_model(system_instruction=system)
        result = await model.generate_content_async(user_prompt)
        return getattr(result, "text", "") or ""
