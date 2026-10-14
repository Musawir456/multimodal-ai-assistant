"""FastAPI app: streaming chat endpoint (Server-Sent Events) plus the web UI.

POST /api/chat  (multipart/form-data)
    session_id : 8-64 chars of [A-Za-z0-9_-], chosen by the client
    message    : optional text
    files      : zero or more attachments (images, PDFs, audio, text/code)
  Response: text/event-stream. Each event is `data: {json}` with one of
    {"type": "text", "text": "..."}   a chunk of the answer
    {"type": "done"}                  the turn finished
    {"type": "error", "text": "..."}  the turn failed

DELETE /api/sessions/{session_id}   forget a conversation
GET    /healthz                      liveness probe
GET    /                             the web UI
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from .assistant import Assistant, AssistantError, Sessions
from .attachments import MAX_UPLOAD_BYTES, AttachmentError, build_block
from .config import Settings
from .llm import LLM, ClaudeLLM
from .transcribe import Transcriber, WhisperTranscriber

log = logging.getLogger("multimodal_assistant")
WEB_DIR = Path(__file__).parent / "web"
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(
    settings: Settings | None = None,
    llm: LLM | None = None,
    transcriber: Transcriber | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    assistant = Assistant(llm or ClaudeLLM(settings), settings)
    transcriber = transcriber or WhisperTranscriber(settings)
    sessions = Sessions(settings.max_sessions)
    app = FastAPI(title="Multi-modal AI Assistant")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.delete("/api/sessions/{session_id}")
    async def reset_session(session_id: str) -> dict:
        if not SESSION_ID.match(session_id):
            raise HTTPException(400, "Invalid session id.")
        return {"cleared": sessions.reset(session_id)}

    @app.post("/api/chat")
    async def chat(
        session_id: Annotated[str, Form()],
        message: Annotated[str, Form()] = "",
        files: Annotated[list[UploadFile] | None, File()] = None,
    ) -> StreamingResponse:
        if not SESSION_ID.match(session_id):
            raise HTTPException(400, "Invalid session id.")
        files = files or []
        if len(files) > settings.max_files_per_message:
            raise HTTPException(400, f"At most {settings.max_files_per_message} files per message.")

        blocks: list[dict] = []
        for f in files:
            data = await f.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"'{f.filename}' is too large.")
            try:
                # audio transcription is blocking, so keep it off the event loop
                blocks.append(await asyncio.to_thread(build_block, f.filename, data, transcriber))
            except AttachmentError as exc:
                raise HTTPException(400, str(exc)) from exc
        if not blocks and not message.strip():
            raise HTTPException(400, "Send a message or attach a file.")

        conv = sessions.get(session_id)

        async def events() -> AsyncIterator[str]:
            async with conv.lock:  # one turn at a time per conversation
                try:
                    async for delta in assistant.reply(conv, message, blocks):
                        yield _sse({"type": "text", "text": delta})
                    yield _sse({"type": "done"})
                except AssistantError as exc:
                    yield _sse({"type": "error", "text": str(exc)})
                except Exception:
                    log.exception("turn failed")
                    yield _sse({"type": "error", "text": "Something went wrong. Try again."})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
