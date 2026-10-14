import base64
import json

from fastapi.testclient import TestClient

from multimodal_assistant.config import Settings
from multimodal_assistant.server import create_app

from .conftest import FakeLLM, FakeTranscriber

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
PDF = b"%PDF-1.7\n" + b"x" * 32
WEBM = b"\x1aE\xdf\xa3" + b"\x00" * 32
SID = "session-abc123"


def make_client(llm=None, **settings):
    llm = llm or FakeLLM()
    app = create_app(Settings(**settings), llm=llm, transcriber=FakeTranscriber())
    return TestClient(app), llm


def events(response):
    out = []
    for chunk in response.text.split("\n\n"):
        if chunk.startswith("data: "):
            out.append(json.loads(chunk[6:]))
    return out


def chat(client, message="", files=(), sid=SID):
    return client.post(
        "/api/chat",
        data={"session_id": sid, "message": message},
        files=[("files", f) for f in files],
    )


def test_healthz_and_index():
    client, _ = make_client()
    assert client.get("/healthz").json() == {"status": "ok"}
    assert "Multi-modal AI Assistant" in client.get("/").text


def test_text_chat_streams_sse_events():
    client, _ = make_client()
    r = chat(client, "hi")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert events(r) == [
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": " world."},
        {"type": "done"},
    ]


def test_image_pdf_and_audio_reach_the_model_as_the_right_blocks():
    client, llm = make_client()
    r = chat(
        client,
        "explain these",
        [
            ("cat.png", PNG, "image/png"),
            ("paper.pdf", PDF, "application/pdf"),
            ("note.webm", WEBM, "audio/webm"),
        ],
    )
    assert events(r)[-1] == {"type": "done"}
    content = llm.calls[0]["messages"][0]["content"]
    assert [b["type"] for b in content] == ["image", "document", "text", "text"]
    assert base64.b64decode(content[0]["source"]["data"]) == PNG
    assert "spoken words" in content[2]["text"]
    assert content[3]["text"] == "explain these"


def test_conversation_memory_is_per_session():
    client, llm = make_client()
    chat(client, "my name is Sam", sid="session-one1")
    chat(client, "what is my name?", sid="session-one1")
    chat(client, "hello", sid="session-two2")
    assert len(llm.calls[1]["messages"]) == 3  # user, assistant, user
    assert len(llm.calls[2]["messages"]) == 1  # a separate, fresh conversation


def test_reset_session_clears_memory():
    client, llm = make_client()
    chat(client, "one")
    assert client.delete(f"/api/sessions/{SID}").json() == {"cleared": True}
    chat(client, "two")
    assert len(llm.calls[1]["messages"]) == 1  # fresh conversation after the reset
    assert client.delete(f"/api/sessions/{SID}").json() == {"cleared": True}
    assert client.delete(f"/api/sessions/{SID}").json() == {"cleared": False}


def test_validation_errors():
    client, llm = make_client(max_files_per_message=2)
    assert chat(client, "hi", sid="bad id!").status_code == 400
    assert chat(client, "   ").status_code == 400
    exe = [("a.exe", b"MZ\x90\x00", "application/octet-stream")]
    assert chat(client, "hi", exe).status_code == 400
    three = [("a.png", PNG, "image/png")] * 3
    assert chat(client, "hi", three).status_code == 400
    assert client.delete("/api/sessions/short").status_code == 400
    assert llm.calls == []


def test_model_failure_is_reported_as_an_error_event_and_leaks_nothing():
    llm = FakeLLM()
    llm.error = RuntimeError("secret internal detail: sk-ant-123")
    client, _ = make_client(llm)
    r = chat(client, "hi")
    evs = events(r)
    assert evs == [{"type": "error", "text": "Something went wrong. Try again."}]
    assert "sk-ant" not in r.text
    # the failed turn is not remembered
    llm.error = None
    chat(client, "again")
    assert len(llm.calls[-1]["messages"]) == 1


def test_a_file_with_a_misleading_name_is_still_classified_by_content():
    client, llm = make_client()
    chat(client, "look", [("totally-a-pdf.pdf", PNG, "application/pdf")])
    assert llm.calls[0]["messages"][0]["content"][0]["type"] == "image"
