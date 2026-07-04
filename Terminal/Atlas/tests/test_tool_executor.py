from typing import Any

import pytest

from app.contracts.abstract_tool import AbstractTool
from app.events.event import Event
from app.events.event_bus import EventBus
from app.tools.tool_executor import ToolExecutor
from app.tools.tool_registry import ToolRegistry
from app.types.tool_result import ToolResult


class SuccessfulTool(AbstractTool):
    @property
    def name(self) -> str:
        return "successful_tool"

    @property
    def description(self) -> str:
        return "A successful test tool."

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult.ok(data=kwargs)


class FailedTool(AbstractTool):
    @property
    def name(self) -> str:
        return "failed_tool"

    @property
    def description(self) -> str:
        return "A test tool that returns a failed result."

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult.fail(error="Tool execution failed.")


class ExceptionTool(AbstractTool):
    @property
    def name(self) -> str:
        return "exception_tool"

    @property
    def description(self) -> str:
        return "A test tool that raises an exception."

    async def execute(self, **kwargs: Any) -> ToolResult:
        raise RuntimeError("Unexpected tool error.")


@pytest.mark.anyio
async def test_tool_executor_executes_successful_tool() -> None:
    event_bus = EventBus()
    tool_registry = ToolRegistry()
    tool_registry.register(SuccessfulTool())

    executor = ToolExecutor(tool_registry, event_bus)

    result = await executor.execute(
        "successful_tool",
        symbol="BTCUSDT",
    )

    assert result.success is True
    assert result.data == {"symbol": "BTCUSDT"}


@pytest.mark.anyio
async def test_tool_executor_handles_failed_result() -> None:
    event_bus = EventBus()
    tool_registry = ToolRegistry()
    tool_registry.register(FailedTool())

    executor = ToolExecutor(tool_registry, event_bus)

    result = await executor.execute("failed_tool")

    assert result.success is False
    assert result.error == "Tool execution failed."


@pytest.mark.anyio
async def test_tool_executor_handles_exception() -> None:
    event_bus = EventBus()
    tool_registry = ToolRegistry()
    tool_registry.register(ExceptionTool())

    executor = ToolExecutor(tool_registry, event_bus)

    result = await executor.execute("exception_tool")

    assert result.success is False
    assert result.error == "Unexpected tool error."


@pytest.mark.anyio
async def test_tool_executor_publishes_lifecycle_events() -> None:
    event_bus = EventBus()
    tool_registry = ToolRegistry()
    tool_registry.register(SuccessfulTool())

    received_events: list[Event] = []

    event_bus.subscribe(
        "tool.started",
        received_events.append,
    )
    event_bus.subscribe(
        "tool.completed",
        received_events.append,
    )

    executor = ToolExecutor(tool_registry, event_bus)

    await executor.execute("successful_tool")

    assert [event.name for event in received_events] == [
        "tool.started",
        "tool.completed",
    ]