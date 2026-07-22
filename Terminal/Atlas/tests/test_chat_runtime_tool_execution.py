from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.memory.conversation import Conversation
from app.models.atlas import AtlasModel
from app.models.atlas.backend import AbstractInferenceBackend
from app.services.chat_runtime import ChatRuntime
from app.services.tool_router import ToolRouter
from app.types.model_result import ModelResult


@dataclass(slots=True)
class FakeToolResult:
    success: bool
    data: Any = None
    error: str | None = None


class RecordingBackend(AbstractInferenceBackend):

    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "recording"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        self.calls += 1
        return ModelResult.ok("MODEL")

    async def generate_from_conversation(
            self,
            conversation: Conversation,
            system_prompt: str | None = None,
            **kwargs: Any,
    ) -> ModelResult:
        self.calls += 1
        return ModelResult.ok("MODEL")


class RecordingToolRouter(ToolRouter):

    def __init__(self, tool_name: str | None) -> None:
        self.tool_name = tool_name

    def route(
            self,
            message: str,
    ) -> str | None:
        return self.tool_name


class RecordingToolExecutor:

    def __init__(self) -> None:
        self.calls = 0
        self.tool_name = None
        self.kwargs = None
        self.result = FakeToolResult(
            success=True,
            data="TOOL OUTPUT",
        )

    async def execute(
            self,
            tool_name: str,
            **kwargs: Any,
    ) -> FakeToolResult:
        self.calls += 1
        self.tool_name = tool_name
        self.kwargs = kwargs
        return self.result


@pytest.mark.anyio
async def test_executes_tool_when_router_returns_tool_name() -> None:
    backend = RecordingBackend()
    executor = RecordingToolExecutor()

    runtime = ChatRuntime(
        AtlasModel(backend),
        tool_router=RecordingToolRouter("system_info"),
        tool_executor=executor,
    )

    await runtime.send(
        Conversation(),
        "show cpu",
    )

    assert executor.calls == 1


@pytest.mark.anyio
async def test_does_not_call_model_when_tool_is_executed() -> None:
    backend = RecordingBackend()
    executor = RecordingToolExecutor()

    runtime = ChatRuntime(
        AtlasModel(backend),
        tool_router=RecordingToolRouter("system_info"),
        tool_executor=executor,
    )

    await runtime.send(
        Conversation(),
        "show cpu",
    )

    assert backend.calls == 0


@pytest.mark.anyio
async def test_passes_tool_name_to_executor() -> None:
    backend = RecordingBackend()
    executor = RecordingToolExecutor()

    runtime = ChatRuntime(
        AtlasModel(backend),
        tool_router=RecordingToolRouter("system_info"),
        tool_executor=executor,
    )

    await runtime.send(
        Conversation(),
        "show cpu",
    )

    assert executor.tool_name == "system_info"


@pytest.mark.anyio
async def test_passes_kwargs_to_executor() -> None:
    backend = RecordingBackend()
    executor = RecordingToolExecutor()

    runtime = ChatRuntime(
        AtlasModel(backend),
        tool_router=RecordingToolRouter("system_info"),
        tool_executor=executor,
    )

    await runtime.send(
        Conversation(),
        "show cpu",
        region="us-east",
        verbose=True,
    )

    assert executor.kwargs == {
        "region": "us-east",
        "verbose": True,
    }


@pytest.mark.anyio
async def test_returns_tool_result_as_model_result() -> None:
    backend = RecordingBackend()
    executor = RecordingToolExecutor()

    runtime = ChatRuntime(
        AtlasModel(backend),
        tool_router=RecordingToolRouter("system_info"),
        tool_executor=executor,
    )

    result = await runtime.send(
        Conversation(),
        "show cpu",
    )

    assert result.success
    assert result.content == "TOOL OUTPUT"