from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class GuardrailVerdict:
    allowed: bool
    reason: str | None = None
    # Texto seguro: entrada saneada ou saída substituta quando bloqueado.
    safe_text: str | None = None


@runtime_checkable
class InputSafetyPolicy(Protocol):
    def evaluate_input(self, text: str) -> GuardrailVerdict:
        ...


@runtime_checkable
class OutputSafetyPolicy(Protocol):
    def evaluate_output(self, text: str) -> GuardrailVerdict:
        ...
