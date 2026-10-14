import asyncio

import pytest

from multimodal_assistant.assistant import (
    OMITTED,
    Assistant,
    AssistantError,
    Conversation,
    Sessions,
    strip_old_media,
)
from multimodal_assistant.config import Settings

IMG = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
PDF = {
    "type": "document",
    "source": {"type": "base64", "media_type": "application/pdf", "data": "B"},
}


def run_reply(assistant, conv, text="", blocks=()):
    async def go():
        return [d async for d in assistant.reply(conv, text, list(blocks))]

    return asyncio.run(go())


def test_reply_streams_and_records_history(llm):
    assistant, conv = Assistant(llm, Settings()), Conversation()
    assert run_reply(assistant, conv, "hi") == ["Hello", " world."]
    assert conv.history == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": "Hello world."},
    ]
    assert llm.calls[0]["system"] == Settings().system_prompt


def test_attachments_come_before_the_text(llm):
    assistant, conv = Assistant(llm, Settings()), Conversation()
    run_reply(assistant, conv, "what is this?", [IMG])
    content = llm.calls[0]["messages"][0]["content"]
    assert [b["type"] for b in content] == ["image", "text"]


def test_attachment_only_message_is_allowed(llm):
    assistant, conv = Assistant(llm, Settings()), Conversation()
    run_reply(assistant, conv, "", [PDF])
    assert [b["type"] for b in llm.calls[0]["messages"][0]["content"]] == ["document"]


def test_empty_message_is_rejected(llm):
    assistant, conv = Assistant(llm, Settings()), Conversation()
    with pytest.raises(AssistantError):
        run_reply(assistant, conv, "   ")
    assert conv.history == [] and llm.calls == []


def test_llm_failure_rolls_back_the_user_message(llm):
    llm.error = RuntimeError("api down")
    assistant, conv = Assistant(llm, Settings()), Conversation()
    with pytest.raises(RuntimeError):
        run_reply(assistant, conv, "hi")
    assert conv.history == []


def test_empty_model_reply_is_an_error_and_rolls_back(llm):
    llm.deltas = ["  "]
    assistant, conv = Assistant(llm, Settings()), Conversation()
    with pytest.raises(AssistantError, match="empty"):
        run_reply(assistant, conv, "hi")
    assert conv.history == []


def test_history_is_trimmed_to_max_turns(llm):
    assistant, conv = Assistant(llm, Settings(max_turns=2)), Conversation()
    for i in range(5):
        run_reply(assistant, conv, f"q{i}")
    assert len(conv.history) == 4
    assert conv.history[0]["role"] == "user"
    assert conv.history[0]["content"][0]["text"] == "q3"


def test_old_media_is_dropped_from_what_the_model_sees_but_kept_in_history(llm):
    assistant, conv = Assistant(llm, Settings(media_turns=1)), Conversation()
    run_reply(assistant, conv, "first", [IMG])
    run_reply(assistant, conv, "second", [PDF])

    seen = llm.calls[1]["messages"]
    assert OMITTED in seen[0]["content"]  # turn 1's image replaced
    assert PDF in seen[2]["content"]  # latest turn keeps its document
    assert IMG in conv.history[0]["content"]  # stored history is untouched


def test_strip_old_media_edge_cases():
    msgs = [
        {"role": "user", "content": [IMG, {"type": "text", "text": "a"}]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [IMG]},
    ]
    assert strip_old_media(msgs, 5) == msgs  # nothing old enough
    none_kept = strip_old_media(msgs, 0)
    assert none_kept[0]["content"][0] == OMITTED and none_kept[2]["content"] == [OMITTED]
    assert msgs[0]["content"][0] == IMG  # input not mutated


def test_sessions_are_isolated_and_lru_evicted():
    sessions = Sessions(max_sessions=2)
    a = sessions.get("a")
    a.history.append({"role": "user", "content": "x"})
    sessions.get("b")
    sessions.get("a")  # touch a so b is the oldest
    sessions.get("c")  # evicts b
    assert len(sessions) == 2
    assert sessions.get("a").history  # a survived
    assert not sessions.get("b").history  # b was evicted and re-created empty
    assert sessions.reset("a") and not sessions.reset("a")
