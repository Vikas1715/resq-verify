from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMResult:
    text: str
    model: str
    model_version: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    thoughts_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_s: float = 0.0
    retry_count: int = 0
    finish_reason: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    tokens_estimated: bool = False

    def usage(self) -> dict:
        return {"input": self.input_tokens, "output": self.output_tokens,
                "thoughts": self.thoughts_tokens, "total": self.total_tokens,
                "estimated": self.tokens_estimated}


class LLMClient(ABC):
    provider: str = "base"
    is_simulated: bool = False
    model: str = ""

    @abstractmethod
    def generate(self, prompt: str, *, system: Optional[str] = None, temperature: Optional[float] = None,
                 max_output_tokens: Optional[int] = None, seed: Optional[int] = None,
                 expect_json: bool = True, response_schema: Optional[dict] = None) -> LLMResult:
        """Return the model's text. Raises LLMCallError after retries are exhausted."""

    def metadata(self) -> dict:
        return {"provider": self.provider, "model": self.model, "is_simulated": self.is_simulated}
