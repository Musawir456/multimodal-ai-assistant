import base64
import struct

import pytest

from multimodal_assistant.attachments import (
    MAX_IMAGE_BYTES,
    MAX_TEXT_CHARS,
    AttachmentError,
    build_block,
    safe_name,
    sniff,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
GIF = b"GIF89a" + b"\x00" * 32
WEBP = b"RIFF" + struct.pack("<I", 100) + b"WEBP" + b"\x00" * 32
PDF = b"%PDF-1.7\n" + b"x" * 32
WAV = b"RIFF" + struct.pack("<I", 100) + b"WAVE" + b"\x00" * 32
WEBM = b"\x1aE\xdf\xa3" + b"\x00" * 32
OGG = b"OggS" + b"\x00" * 32
MP3 = b"ID3" + b"\x00" * 32
M4A = b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 32


class FakeTranscriber:
    def __init__(self, text="hello from audio"):
        self.text, self.calls = text, []

    def transcribe(self, audio):
        self.calls.append(audio)
        return self.text


@pytest.mark.parametrize(
    "data, expected",
    [
        (PNG, ("image", "image/png")),
        (JPEG, ("image", "image/jpeg")),
        (GIF, ("image", "image/gif")),
        (WEBP, ("image", "image/webp")),
        (PDF, ("pdf", "application/pdf")),
        (WAV, ("audio", "audio/wav")),
        (WEBM, ("audio", "audio/webm")),
        (OGG, ("audio", "audio/ogg")),
        (MP3, ("audio", "audio/mpeg")),
        (M4A, ("audio", "audio/mp4")),
        (b"just some text", None),
    ],
)
def test_sniff(data, expected):
    assert sniff(data) == expected


def test_image_becomes_base64_image_block():
    block = build_block("cat.png", PNG)
    assert block["type"] == "image"
    assert block["source"]["media_type"] == "image/png"
    assert base64.b64decode(block["source"]["data"]) == PNG


def test_pdf_becomes_document_block():
    block = build_block("report.pdf", PDF)
    assert block["type"] == "document"
    assert block["source"]["media_type"] == "application/pdf"


def test_type_comes_from_bytes_not_the_file_name():
    assert build_block("holiday.pdf", PNG)["type"] == "image"
    assert build_block("photo.png", PDF)["type"] == "document"
    assert build_block("notes.txt", PNG)["type"] == "image"


def test_audio_is_transcribed_into_a_text_block():
    tr = FakeTranscriber()
    block = build_block("memo.webm", WEBM, tr)
    assert block["type"] == "text"
    assert "hello from audio" in block["text"] and "memo.webm" in block["text"]
    assert tr.calls == [WEBM]


def test_silent_audio_is_reported_not_an_error():
    block = build_block("quiet.wav", WAV, FakeTranscriber(""))
    assert "no recognizable speech" in block["text"]


def test_audio_without_a_transcriber_is_rejected():
    with pytest.raises(AttachmentError, match="not enabled"):
        build_block("memo.mp3", MP3, None)


def test_text_and_code_files_are_inlined():
    block = build_block("data.csv", b"a,b\n1,2\n")
    assert block == {"type": "text", "text": "[File 'data.csv']\na,b\n1,2\n"}


def test_long_text_is_truncated():
    block = build_block("big.txt", b"x" * (MAX_TEXT_CHARS + 500))
    assert "truncated" in block["text"]
    assert len(block["text"]) < MAX_TEXT_CHARS + 100


def test_non_utf8_text_is_rejected():
    with pytest.raises(AttachmentError, match="UTF-8"):
        build_block("bad.txt", b"\xff\xfe\x00\x00 broken")


@pytest.mark.parametrize("name, data", [("x.exe", b"MZ\x90\x00"), ("noext", b"hello")])
def test_unsupported_types_are_rejected(name, data):
    with pytest.raises(AttachmentError, match="Unsupported"):
        build_block(name, data)


def test_empty_file_is_rejected():
    with pytest.raises(AttachmentError, match="empty"):
        build_block("a.png", b"")


def test_oversized_image_is_rejected():
    with pytest.raises(AttachmentError, match="too large"):
        build_block("huge.png", PNG + b"\x00" * MAX_IMAGE_BYTES)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\photo.png", "photo.png"),
        ("evil\x00name\n.txt", "evilname.txt"),
        ("", "file"),
        (None, "file"),
        ("a" * 300 + ".txt", ("a" * 300 + ".txt")[:80]),
    ],
)
def test_safe_name(raw, expected):
    assert safe_name(raw) == expected
