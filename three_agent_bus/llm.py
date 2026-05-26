from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Union


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


class ClaudeCodeLLM(LLM):
    """Powered by Claude Code. Each call invokes `claude -p` as a subprocess.

    Uses the local Claude Code authentication (OAuth, keychain, or whatever the
    host has configured), so no ANTHROPIC_API_KEY env var is required when
    `claude` is set up locally. Slower per call than the direct API path
    (subprocess + agentic harness overhead), but it doesn't bill against an API
    key — it draws from whatever subscription powers Claude Code.
    """

    def __init__(
        self,
        claude_binary: str = "claude",
        model: Optional[str] = None,
        max_turns: int = 2,
        timeout_seconds: float = 180.0,
        disable_tools: bool = True,
        replace_system_prompt: bool = True,
        override_settings: bool = True,
        extra_args: Optional[list[str]] = None,
    ) -> None:
        self._binary = claude_binary
        self._model = model
        self._max_turns = max_turns
        self._timeout = timeout_seconds
        self._disable_tools = disable_tools
        self._replace_system_prompt = replace_system_prompt
        self._override_settings = override_settings
        self._extra_args = list(extra_args or [])

    def call(self, system: str, user: str) -> LLMResponse:
        system_flag = "--system-prompt" if self._replace_system_prompt else "--append-system-prompt"
        cmd = [
            self._binary,
            "-p", user,
            system_flag, system,
            "--max-turns", str(self._max_turns),
            "--output-format", "text",
        ]
        if self._disable_tools:
            cmd.extend(["--tools", ""])
        if self._override_settings:
            cmd.extend(["--setting-sources", ""])
        if self._model is not None:
            cmd.extend(["--model", self._model])
        cmd.extend(self._extra_args)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"claude timed out after {self._timeout}s"
            ) from e
        if result.returncode != 0:
            raise RuntimeError(
                f"claude exited {result.returncode}: {result.stderr.strip()[:500]}"
            )
        return LLMResponse(text=result.stdout)
