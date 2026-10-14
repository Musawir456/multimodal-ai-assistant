"""LLM abstraction (async streaming)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from .config import Settings


class LLM(Protocol):
    def stream(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        """Yield text deltas for the assistant reply to `messages` (Anthropic block format)."""
        ...


class ClaudeLLM:
    """Anthropic backend. Requires ANTHROPIC_API_KEY in the environment."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic()
        return self._client

    async def stream(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        async with self._get_client().messages.stream(
            model=self._settings.anthropic_model,
            max_tokens=self._settings.max_tokens,
            system=system,
            messages=messages,
        ) as stream:
            async for delta in stream.text_stream:
                yield delta
