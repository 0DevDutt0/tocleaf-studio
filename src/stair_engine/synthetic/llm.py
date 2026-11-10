"""OpenAI-compatible chat client for synthetic question generation.

Configuration comes only from environment variables, never from source files:

  STAIR_LLM_API_BASE   base URL, for example http://127.0.0.1:11434/v1 for a local Ollama
  STAIR_LLM_API_KEY    API key; local servers that ignore keys still need a non-empty value
  STAIR_LLM_MODEL      model name as the server knows it

The same variable names as the upstream STAIR CLI are used on purpose, so one set of
settings works for both. Provider selection is independent of Claude Code.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol


class LLMConfigurationError(RuntimeError):
    """Raised when no endpoint is configured. The engine never fabricates questions."""


class ChatClient(Protocol):
    model: str

    def complete(self, messages: list[dict]) -> str: ...


@dataclass
class OpenAICompatibleClient:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 300.0
    max_attempts: int = 4
    temperature: float = 0.3
    max_output_tokens: int = 1500
    extra_body: dict = field(default_factory=dict)
    sleep: Callable[[float], None] = time.sleep
    _client: object | None = field(default=None, repr=False)

    def _openai(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout_seconds,
                max_retries=0,  # retries are handled here so every attempt is logged
            )
        return self._client

    def complete(self, messages: list[dict]) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._openai().chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                    extra_body=self.extra_body or None,
                )
                content = response.choices[0].message.content
                return content or ""
            except Exception as error:  # noqa: BLE001 - classified below, then re-raised if final
                last_error = error
                retryable = _is_retryable(error)
                if not retryable or attempt == self.max_attempts:
                    raise
                self.sleep(min(30.0, 2.0 ** attempt))
        raise RuntimeError(f"unreachable: {last_error}")


def _is_retryable(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status in (408, 409, 429) or (isinstance(status, int) and status >= 500):
        return True
    name = type(error).__name__
    return name in {"APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError"}


def client_from_environment(
    *,
    extra_body: dict | None = None,
    timeout_seconds: float = 300.0,
    temperature: float = 0.3,
) -> OpenAICompatibleClient:
    base_url = os.environ.get("STAIR_LLM_API_BASE", "").strip()
    api_key = os.environ.get("STAIR_LLM_API_KEY", "").strip()
    model = os.environ.get("STAIR_LLM_MODEL", "").strip()
    missing = [
        name
        for name, value in (
            ("STAIR_LLM_API_BASE", base_url),
            ("STAIR_LLM_API_KEY", api_key),
            ("STAIR_LLM_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise LLMConfigurationError(
            "Synthetic question generation needs an OpenAI-compatible endpoint. Set "
            + ", ".join(missing)
            + " in the environment. No questions were generated."
        )
    return OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        extra_body=dict(extra_body or {}),
    )
