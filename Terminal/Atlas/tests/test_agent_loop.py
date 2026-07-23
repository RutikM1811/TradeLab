from __future__ import annotations

from typing import Any

import pytest

from app.memory.conversation import Conversation
from app.services.agent_loop import AgentLoop
from app.tools.tool_schema import ToolSchema
from app.types.model_result import ModelResult


class RecordingModel:

    def __init__(self) -> None:
        self.calls = 0
        self.prompt: str | None = None
        self.kwargs: dict[str, Any] | None = None

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        self.calls += 1
        self.prompt = prompt
        self.kwargs = kwargs
        return ModelResult.ok("Hello!")


class RecordingPromptBuilder:

    def __init__(self) -> None:
        self.calls = 0
        self.user_message: str | None = None
        self.tools: tuple[ToolSchema, ...] | None = None

    def build(
            self,
            user_message: str,
            tools: tuple[ToolSchema, ...],
    ) -> str:
        self.calls += 1
        self.user_message = user_message
        self.tools = tools
        return f"PROMPT::{user_message}"


class RecordingParser:

    def __init__(self) -> None:
        self.calls = 0
        self.response: str | None = None

    def parse(
            self,
            response: str,
    ):
        self.calls += 1
        self.response = response
        return None


class FakeTool:

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            "system_info",
            "Returns system information.",
            {},
        )


class FakeRegistry:

    def values(self):
        return (
            FakeTool(),
        )


@pytest.mark.anyio
async def test_adds_user_message() -> None:
    conversation = Conversation()

    loop = AgentLoop(
        RecordingModel(),
        FakeRegistry(),
        RecordingPromptBuilder(),
        RecordingParser(),
    )

    await loop.run(
        conversation,
        "Hello",
    )

    assert len(conversation) == 1


@pytest.mark.anyio
async def test_calls_prompt_builder() -> None:
    builder = RecordingPromptBuilder()

    loop = AgentLoop(
        RecordingModel(),
        FakeRegistry(),
        builder,
        RecordingParser(),
    )

    await loop.run(
        Conversation(),
        "Hello",
    )

    assert builder.calls == 1


@pytest.mark.anyio
async def test_calls_model() -> None:
    model = RecordingModel()

    loop = AgentLoop(
        model,
        FakeRegistry(),
        RecordingPromptBuilder(),
        RecordingParser(),
    )

    await loop.run(
        Conversation(),
        "Hello",
    )

    assert model.calls == 1


@pytest.mark.anyio
async def test_calls_parser() -> None:
    parser = RecordingParser()

    loop = AgentLoop(
        RecordingModel(),
        FakeRegistry(),
        RecordingPromptBuilder(),
        parser,
    )

    await loop.run(
        Conversation(),
        "Hello",
    )

    assert parser.calls == 1


@pytest.mark.anyio
async def test_returns_model_result_when_no_tool() -> None:
    loop = AgentLoop(
        RecordingModel(),
        FakeRegistry(),
        RecordingPromptBuilder(),
        RecordingParser(),
    )

    result = await loop.run(
        Conversation(),
        "Hello",
    )

    assert result.success
    assert result.content == "Hello!"


@pytest.mark.anyio
async def test_preserves_conversation() -> None:
    conversation = Conversation()

    loop = AgentLoop(
        RecordingModel(),
        FakeRegistry(),
        RecordingPromptBuilder(),
        RecordingParser(),
    )

    await loop.run(
        conversation,
        "Hello",
    )

    assert len(conversation) == 1


@pytest.mark.anyio
async def test_passes_kwargs_to_model() -> None:
    model = RecordingModel()

    loop = AgentLoop(
        model,
        FakeRegistry(),
        RecordingPromptBuilder(),
        RecordingParser(),
    )

    await loop.run(
        Conversation(),
        "Hello",
        temperature=0.5,
        max_tokens=100,
    )

    assert model.kwargs == {
        "temperature": 0.5,
        "max_tokens": 100,
    }


@pytest.mark.anyio
async def test_handles_empty_message() -> None:
    loop = AgentLoop(
        RecordingModel(),
        FakeRegistry(),
        RecordingPromptBuilder(),
        RecordingParser(),
    )

    result = await loop.run(
        Conversation(),
        "",
    )

    assert result.success


@pytest.mark.anyio
async def test_handles_unicode() -> None:
    loop = AgentLoop(
        RecordingModel(),
        FakeRegistry(),
        RecordingPromptBuilder(),
        RecordingParser(),
    )

    result = await loop.run(
        Conversation(),
        "नमस्कार 世界 🚀",
    )

    assert result.success


@pytest.mark.anyio
async def test_prompt_contains_user_message() -> None:
    model = RecordingModel()

    loop = AgentLoop(
        model,
        FakeRegistry(),
        RecordingPromptBuilder(),
        RecordingParser(),
    )

    await loop.run(
        Conversation(),
        "Hello Atlas",
    )

    assert "Hello Atlas" in model.prompt