from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Union


@dataclass
class LLMResponse:
    text: str


class LLM(ABC):
    @abstractmethod
    def call(self, system: str, user: str) -> LLMResponse:
        ...


ScriptedResponse = Union[str, Callable[[str, str], str]]


class ScriptedLLM(LLM):
    def __init__(self, responses: list[ScriptedResponse]) -> None:
        self._responses = list(responses)
        self._index = 0

    def call(self, system: str, user: str) -> LLMResponse:
        if self._index >= len(self._responses):
            raise IndexError("ScriptedLLM exhausted")
        r = self._responses[self._index]
        self._index += 1
        text = r(system, user) if callable(r) else r
        return LLMResponse(text=text)


class AnthropicLLM(LLM):
    def __init__(
        self,
        model: str = "claude-opus-4-7",
        max_tokens: int = 16000,
        effort: str = "high",
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic SDK not installed; run `pip install anthropic`"
            ) from e
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens
        self._effort = effort

    def call(self, system: str, user: str) -> LLMResponse:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": self._effort},
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        return LLMResponse(text=text)
