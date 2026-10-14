"""Command line: `multimodal-assistant serve` and `multimodal-assistant ask`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from .assistant import Assistant, AssistantError, Conversation
from .attachments import AttachmentError, build_block
from .config import Settings
from .llm import ClaudeLLM
from .transcribe import WhisperTranscriber


def _need_key() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    print("Error: set the ANTHROPIC_API_KEY environment variable.", file=sys.stderr)
    return False


def _serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from .server import create_app

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(settings), host=args.host, port=args.port)
    return 0


def _ask(args: argparse.Namespace, settings: Settings) -> int:
    transcriber = WhisperTranscriber(settings)
    blocks = []
    for path in args.file:
        try:
            with open(path, "rb") as fh:
                blocks.append(build_block(os.path.basename(path), fh.read(), transcriber))
        except OSError as exc:
            print(f"Error: cannot read {path}: {exc.strerror}", file=sys.stderr)
            return 2
        except AttachmentError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    async def run() -> None:
        assistant = Assistant(ClaudeLLM(settings), settings)
        async for delta in assistant.reply(Conversation(), args.question, blocks):
            print(delta, end="", flush=True)
        print()

    try:
        asyncio.run(run())
    except AssistantError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        prog="multimodal-assistant",
        description="Chat with Claude using text, images, PDFs and audio.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the web UI and API")
    serve.add_argument("--host", default=settings.host)
    serve.add_argument("--port", type=int, default=settings.port)
    serve.set_defaults(func=_serve)

    ask = sub.add_parser("ask", help="Ask one question, optionally about files")
    ask.add_argument("question")
    ask.add_argument("-f", "--file", action="append", default=[], help="Attach a file (repeatable)")
    ask.set_defaults(func=_ask)

    args = parser.parse_args(argv)
    if not _need_key():
        return 2
    return args.func(args, settings)


if __name__ == "__main__":
    raise SystemExit(main())
