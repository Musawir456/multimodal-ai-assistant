"""Turn uploaded files into Claude content blocks.

File types are detected from the file's leading bytes ("magic numbers"), never from its
name or the browser-supplied content type, so a renamed or mislabeled file cannot slip
through as something else.
"""

from __future__ import annotations

import base64
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .transcribe import Transcriber

MB = 1024 * 1024
MAX_IMAGE_BYTES = 5 * MB  # Claude's per-image limit
MAX_PDF_BYTES = 20 * MB
MAX_AUDIO_BYTES = 25 * MB
MAX_TEXT_BYTES = 1 * MB
MAX_TEXT_CHARS = 60_000
MAX_UPLOAD_BYTES = MAX_AUDIO_BYTES  # hard cap for reading any single upload into memory

TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml", ".log", ".ini", ".toml",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".java", ".c", ".cpp", ".h", ".go",
    ".rs", ".rb", ".php", ".sh", ".sql",
}


class AttachmentError(ValueError):
    """The file cannot be used; the message is safe to show to the user."""


def safe_name(name: str | None) -> str:
    """Strip any directory part and control characters from a user-supplied file name."""
    base = os.path.basename((name or "").replace("\\", "/"))
    base = re.sub(r"[\x00-\x1f\x7f]", "", base).strip()
    return base[:80] or "file"


def sniff(data: bytes) -> tuple[str, str] | None:
    """Return (kind, media_type) from magic bytes, or None if unrecognized.

    kind is one of: "image", "pdf", "audio".
    """
    head = data[:16]
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image", "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image", "image/jpeg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image", "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image", "image/webp"
    if b"%PDF-" in data[:1024]:
        return "pdf", "application/pdf"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "audio", "audio/wav"
    if head.startswith(b"\x1aE\xdf\xa3"):
        return "audio", "audio/webm"
    if head.startswith(b"OggS"):
        return "audio", "audio/ogg"
    if head.startswith(b"fLaC"):
        return "audio", "audio/flac"
    if head.startswith(b"ID3") or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio", "audio/mpeg"
    if head[4:8] == b"ftyp":
        return "audio", "audio/mp4"
    return None


def _b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("ascii")


def _check_size(label: str, name: str, size: int, limit: int) -> None:
    if size > limit:
        raise AttachmentError(
            f"{label} '{name}' is too large ({size / MB:.1f} MB; the limit is {limit // MB} MB)."
        )


def build_block(name: str | None, data: bytes, transcriber: Transcriber | None = None) -> dict:
    """Convert one uploaded file into a Claude content block."""
    name = safe_name(name)
    if not data:
        raise AttachmentError(f"'{name}' is empty.")

    detected = sniff(data)
    if detected:
        kind, media_type = detected
        if kind == "image":
            _check_size("Image", name, len(data), MAX_IMAGE_BYTES)
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": _b64(data)},
            }
        if kind == "pdf":
            _check_size("PDF", name, len(data), MAX_PDF_BYTES)
            return {
                "type": "document",
                "source": {"type": "base64", "media_type": media_type, "data": _b64(data)},
            }
        # audio
        _check_size("Audio file", name, len(data), MAX_AUDIO_BYTES)
        if transcriber is None:
            raise AttachmentError("Audio attachments are not enabled on this server.")
        transcript = transcriber.transcribe(data)
        if not transcript:
            return {
                "type": "text",
                "text": f"[Audio file '{name}' contained no recognizable speech.]",
            }
        return {"type": "text", "text": f"[Transcript of audio file '{name}']\n{transcript}"}

    ext = os.path.splitext(name)[1].lower()
    if ext in TEXT_EXTENSIONS:
        _check_size("Text file", name, len(data), MAX_TEXT_BYTES)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AttachmentError(f"'{name}' is not valid UTF-8 text.") from exc
        note = ""
        if len(text) > MAX_TEXT_CHARS:
            text, note = text[:MAX_TEXT_CHARS], "\n[...file truncated...]"
        return {"type": "text", "text": f"[File '{name}']\n{text}{note}"}

    raise AttachmentError(
        f"Unsupported file type: '{name}'. Use PNG/JPEG/GIF/WebP images, PDFs, audio "
        "(WAV/MP3/OGG/WebM/FLAC/M4A) or text/code files."
    )
