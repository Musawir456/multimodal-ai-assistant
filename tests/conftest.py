import copy

import pytest


class FakeLLM:
    """Async streaming fake. Records a deep copy of every `messages` payload it receives."""

    def __init__(self, deltas=("Hello", " world.")):
        self.deltas = list(deltas)
        self.calls = []
        self.error = None

    async def stream(self, system, messages):
        self.calls.append({"system": system, "messages": copy.deepcopy(messages)})
        if self.error:
            raise self.error
        for d in self.deltas:
            yield d


class FakeTranscriber:
    def __init__(self, text="spoken words"):
        self.text = text

    def transcribe(self, audio):
        return self.text


@pytest.fixture
def llm():
    return FakeLLM()
