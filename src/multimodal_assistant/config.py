"""Runtime configuration, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful multi-modal assistant. The user may attach images, PDF documents, "
    "text files, or audio (audio arrives as a transcript). Look carefully at every attachment "
    "and answer from its actual content; say so plainly when something is unreadable or "
    "missing instead of guessing. Be clear and concise, and reply in the user's language."
)


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8000
    anthropic_model: str = "claude-sonnet-5-5"
    max_tokens: int = 1500
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_turns: int = 10  # conversation turns kept in memory
    media_turns: int = 2  # only the last N user turns keep their images/PDFs
    max_sessions: int = 100
    max_files_per_message: int = 5
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        d = cls()
        return cls(
            host=env("HOST", d.host),
            port=int(env("PORT", d.port)),
            anthropic_model=env("ANTHROPIC_MODEL", d.anthropic_model),
            max_tokens=int(env("MAX_TOKENS", d.max_tokens)),
            system_prompt=env("SYSTEM_PROMPT", d.system_prompt),
            max_turns=int(env("MAX_TURNS", d.max_turns)),
            media_turns=int(env("MEDIA_TURNS", d.media_turns)),
            max_sessions=int(env("MAX_SESSIONS", d.max_sessions)),
            max_files_per_message=int(env("MAX_FILES_PER_MESSAGE", d.max_files_per_message)),
            whisper_model=env("WHISPER_MODEL", d.whisper_model),
            whisper_device=env("WHISPER_DEVICE", d.whisper_device),
            whisper_compute_type=env("WHISPER_COMPUTE_TYPE", d.whisper_compute_type),
        )
