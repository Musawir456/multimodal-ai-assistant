"""Speech-to-text for audio attachments (optional: `pip install ".[audio]"`)."""

from __future__ import annotations

import io
from typing import Protocol

from .attachments import AttachmentError
from .config import Settings


class Transcriber(Protocol):
    def transcribe(self, audio: bytes) -> str:
        """Blocking transcription of an encoded audio clip. Returns '' if no speech."""
        ...


class WhisperTranscriber:
    """faster-whisper backend. The model is loaded lazily on first use."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise AttachmentError(
                    "Audio support is not installed. Run: pip install \".[audio]\""
                ) from exc
            self._model = WhisperModel(
                self._settings.whisper_model,
                device=self._settings.whisper_device,
                compute_type=self._settings.whisper_compute_type,
            )
        return self._model

    def transcribe(self, audio: bytes) -> str:
        segments, _info = self._load().transcribe(io.BytesIO(audio), vad_filter=True, beam_size=1)
        return " ".join(seg.text.strip() for seg in segments).strip()
