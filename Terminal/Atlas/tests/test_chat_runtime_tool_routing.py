from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from app.memory.conversation import Conversation
from app.models.atlas import AtlasModel
from app.models.atlas.backend import AbstractInferenceBackend
from app.services.chat_runtime import ChatRuntime
from app.services.tool_router import ToolRouter
from app.types.model_result import ModelResult

class RecordingBackend(AbstractInferenceBackend):

    @property
    def name(self) -> str:
        return "recording"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        return ModelResult.ok("Hello!")

    async def generate_stream(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> AsyncIterator[str]:
        yield "Hello!"

    async def generate_from_conversation(
            self,
            conversation: Conversation,
            system_prompt: str | None = None,
            **kwargs: Any,
    ) -> ModelResult:
        return ModelResult.ok("Hello!")

    async def generate_stream_from_conversation(
            self,
            conversation: Conversation,
            system_prompt: str | None = None,
            **kwargs: Any,
    ) -> AsyncIterator[str]:
        yield "Hello!"


class RecordingToolRouter(ToolRouter):

    def __init__(self) -> None:
        self.calls = 0
        self.last_message: str | None = None
        self.result: str | None = None

    def route(
            self,
            message: str,
    ) -> str | None:
        self.calls += 1
        self.last_message = message
        return self.result


@pytest.mark.anyio
async def test_chat_runtime_passes_user_message_to_tool_router() -> None:
    router = RecordingToolRouter()

    runtime = ChatRuntime(
        AtlasModel(RecordingBackend()),
        tool_router=router,
    )

    await runtime.send(
        Conversation(),
        "Hello Atlas",
    )

    assert router.last_message == "Hello Atlas"


@pytest.mark.anyio
async def test_chat_runtime_calls_router_once_per_message() -> None:
    router = RecordingToolRouter()

    runtime = ChatRuntime(
        AtlasModel(RecordingBackend()),
        tool_router=router,
    )

    await runtime.send(
        Conversation(),
        "Hello",
    )

    assert router.calls == 1


@pytest.mark.anyio
async def test_chat_runtime_routes_each_message_independently() -> None:
    router = RecordingToolRouter()

    runtime = ChatRuntime(
        AtlasModel(RecordingBackend()),
        tool_router=router,
    )

    conversation = Conversation()

    await runtime.send(
        conversation,
        "First",
    )

    await runtime.send(
        conversation,
        "Second",
    )

    assert router.calls == 2
    assert router.last_message == "Second"


@pytest.mark.anyio
async def test_chat_runtime_continues_when_router_returns_none() -> None:
    router = RecordingToolRouter()

    runtime = ChatRuntime(
        AtlasModel(RecordingBackend()),
        tool_router=router,
    )

    result = await runtime.send(
        Conversation(),
        "Hello",
    )

    assert result.success
    assert result.content == "Hello!"


@pytest.mark.anyio
async def test_chat_runtime_passes_original_message_without_modification() -> None:
    router = RecordingToolRouter()

    runtime = ChatRuntime(
        AtlasModel(RecordingBackend()),
        tool_router=router,
    )

    await runtime.send(
        Conversation(),
        "SHOW MY CPU",
    )

    assert router.last_message == "SHOW MY CPU"