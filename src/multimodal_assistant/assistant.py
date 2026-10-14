"""Conversation state and the core "ask" turn."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from .config import Settings
from .llm import LLM

MEDIA_BLOCK_TYPES = {"image", "document"}
OMITTED = {"type": "text", "text": "[earlier attachment omitted to save tokens]"}


class AssistantError(RuntimeError):
    """The turn failed in a way the user should hear about."""


@dataclass
class Conversation:
    history: list[dict] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class Sessions:
    """In-memory conversations, evicting the least recently used when full."""

    def __init__(self, max_sessions: int = 100) -> None:
        self._max = max(1, max_sessions)
        self._items: OrderedDict[str, Conversation] = OrderedDict()

    def get(self, session_id: str) -> Conversation:
        conv = self._items.get(session_id)
        if conv is None:
            conv = self._items[session_id] = Conversation()
            while len(self._items) > self._max:
                self._items.popitem(last=False)
        else:
            self._items.move_to_end(session_id)
        return conv

    def reset(self, session_id: str) -> bool:
        return self._items.pop(session_id, None) is not None

    def __len__(self) -> int:
        return len(self._items)


def strip_old_media(messages: list[dict], keep_last_user_turns: int) -> list[dict]:
    """Replace images/PDFs in all but the last N user turns with a short placeholder.

    Media is by far the most expensive part of a long chat; the model's own earlier answers
    already carry what mattered. The stored history is never modified.
    """
    user_idx = [i for i, m in enumerate(messages) if m["role"] == "user"]
    old = set(user_idx) if keep_last_user_turns <= 0 else set(user_idx[:-keep_last_user_turns])
    out = []
    for i, m in enumerate(messages):
        if i in old and isinstance(m["content"], list):
            blocks = [OMITTED if b.get("type") in MEDIA_BLOCK_TYPES else b for b in m["content"]]
            m = {**m, "content": blocks}
        out.append(m)
    return out


class Assistant:
    def __init__(self, llm: LLM, settings: Settings) -> None:
        self._llm = llm
        self._settings = settings

    async def reply(
        self, conv: Conversation, text: str, blocks: list[dict]
    ) -> AsyncIterator[str]:
        """Add the user's message (attachments first, then text) and stream the answer."""
        content = list(blocks)
        if text.strip():
            content.append({"type": "text", "text": text.strip()})
        if not content:
            raise AssistantError("Send a message or attach a file.")

        conv.history.append({"role": "user", "content": content})
        parts: list[str] = []
        try:
            prepared = strip_old_media(conv.history, self._settings.media_turns)
            async for delta in self._llm.stream(self._settings.system_prompt, prepared):
                parts.append(delta)
                yield delta
        except BaseException:
            conv.history.pop()  # keep roles alternating; drop the unanswered message
            raise

        answer = "".join(parts).strip()
        if not answer:
            conv.history.pop()
            raise AssistantError("The model returned an empty reply.")
        conv.history.append({"role": "assistant", "content": answer})
        limit = max(2, self._settings.max_turns * 2)
        if len(conv.history) > limit:
            conv.history[:] = conv.history[-limit:]
