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


class KwargsTool(AbstractTool):
    def __init__(self) -> None:
        self.received_kwargs: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "kwargs_tool"

    @property
    def description(self) -> str:
        return "Captures kwargs."

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.received_kwargs = kwargs
        return ToolResult.ok(data="done")


@pytest.mark.anyio
async def test_execute_without_kwargs() -> None:
    bus = EventBus()
    registry = ToolRegistry()
    registry.register(SuccessfulTool())

    executor = ToolExecutor(registry, bus)

    result = await executor.execute("successful_tool")

    assert result.success


@pytest.mark.anyio
async def test_execute_with_multiple_kwargs() -> None:
    bus = EventBus()
    registry = ToolRegistry()

    tool = KwargsTool()
    registry.register(tool)

    executor = ToolExecutor(registry, bus)

    await executor.execute(
        "kwargs_tool",
        symbol="BTC",
        timeframe="1h",
        limit=100,
    )

    assert tool.received_kwargs == {
        "symbol": "BTC",
        "timeframe": "1h",
        "limit": 100,
    }


@pytest.mark.anyio
async def test_started_event_contains_tool_name() -> None:
    bus = EventBus()
    registry = ToolRegistry()
    registry.register(SuccessfulTool())

    events = []

    bus.subscribe(
        "tool.started",
        events.append,
    )

    executor = ToolExecutor(registry, bus)

    await executor.execute("successful_tool")

    assert events[0].payload["tool_name"] == "successful_tool"


@pytest.mark.anyio
async def test_completed_event_contains_tool_name() -> None:
    bus = EventBus()
    registry = ToolRegistry()
    registry.register(SuccessfulTool())

    events = []

    bus.subscribe(
        "tool.completed",
        events.append,
    )

    executor = ToolExecutor(registry, bus)

    await executor.execute("successful_tool")

    assert events[0].payload["tool_name"] == "successful_tool"


@pytest.mark.anyio
async def test_completed_event_success_true() -> None:
    bus = EventBus()
    registry = ToolRegistry()
    registry.register(SuccessfulTool())

    events = []

    bus.subscribe(
        "tool.completed",
        events.append,
    )

    executor = ToolExecutor(registry, bus)

    await executor.execute("successful_tool")

    assert events[0].payload["success"] is True


@pytest.mark.anyio
async def test_failed_result_publishes_failed_event() -> None:
    bus = EventBus()
    registry = ToolRegistry()
    registry.register(FailedTool())

    events = []

    bus.subscribe(
        "tool.failed",
        events.append,
    )

    executor = ToolExecutor(registry, bus)

    await executor.execute("failed_tool")

    assert len(events) == 1


@pytest.mark.anyio
async def test_failed_event_contains_tool_name() -> None:
    bus = EventBus()
    registry = ToolRegistry()
    registry.register(FailedTool())

    events = []

    bus.subscribe(
        "tool.failed",
        events.append,
    )

    executor = ToolExecutor(registry, bus)

    await executor.execute("failed_tool")

    assert events[0].payload["tool_name"] == "failed_tool"


@pytest.mark.anyio
async def test_failed_event_success_false() -> None:
    bus = EventBus()
    registry = ToolRegistry()
    registry.register(FailedTool())

    events = []

    bus.subscribe(
        "tool.failed",
        events.append,
    )

    executor = ToolExecutor(registry, bus)

    await executor.execute("failed_tool")

    assert events[0].payload["success"] is False


@pytest.mark.anyio
async def test_exception_event_contains_error() -> None:
    bus = EventBus()
    registry = ToolRegistry()
    registry.register(ExceptionTool())

    events = []

    bus.subscribe(
        "tool.failed",
        events.append,
    )

    executor = ToolExecutor(registry, bus)

    await executor.execute("exception_tool")

    assert events[0].payload["error"] == "Unexpected tool error."


@pytest.mark.anyio
async def test_exception_event_contains_tool_name() -> None:
    bus = EventBus()
    registry = ToolRegistry()
    registry.register(ExceptionTool())

    events = []

    bus.subscribe(
        "tool.failed",
        events.append,
    )

    executor = ToolExecutor(registry, bus)

    await executor.execute("exception_tool")

    assert events[0].payload["tool_name"] == "exception_tool"


@pytest.mark.anyio
async def test_result_data_is_preserved() -> None:
    bus = EventBus()
    registry = ToolRegistry()
    registry.register(SuccessfulTool())

    executor = ToolExecutor(registry, bus)

    result = await executor.execute(
        "successful_tool",
        value=10,
    )

    assert result.data == {"value": 10}


@pytest.mark.anyio
async def test_result_error_is_preserved() -> None:
    bus = EventBus()
    registry = ToolRegistry()
    registry.register(FailedTool())

    executor = ToolExecutor(registry, bus)

    result = await executor.execute("failed_tool")

    assert result.error == "Tool execution failed."


@pytest.mark.anyio
async def test_multiple_successful_executions() -> None:
    bus = EventBus()
    registry = ToolRegistry()
    registry.register(SuccessfulTool())

    executor = ToolExecutor(registry, bus)

    for _ in range(5):
        result = await executor.execute("successful_tool")
        assert result.success


@pytest.mark.anyio
async def test_multiple_failed_executions() -> None:
    bus = EventBus()
    registry = ToolRegistry()
    registry.register(FailedTool())

    executor = ToolExecutor(registry, bus)

    for _ in range(3):
        result = await executor.execute("failed_tool")
        assert not result.success


@pytest.mark.anyio
async def test_exception_execution_multiple_times() -> None:
    bus = EventBus()
    registry = ToolRegistry()
    registry.register(ExceptionTool())

    executor = ToolExecutor(registry, bus)

    for _ in range(2):
        result = await executor.execute("exception_tool")
        assert not result.success


@pytest.mark.anyio
async def test_execute_accepts_unicode_kwargs() -> None:
    bus = EventBus()
    registry = ToolRegistry()

    tool = KwargsTool()
    registry.register(tool)

    executor = ToolExecutor(registry, bus)

    await executor.execute(
        "kwargs_tool",
        text="नमस्कार 🚀",
    )

    assert tool.received_kwargs["text"] == "नमस्कार 🚀"


@pytest.mark.anyio
async def test_execute_accepts_empty_kwargs() -> None:
    bus = EventBus()
    registry = ToolRegistry()

    tool = KwargsTool()
    registry.register(tool)

    executor = ToolExecutor(registry, bus)

    await executor.execute("kwargs_tool")

    assert tool.received_kwargs == {}


@pytest.mark.anyio
async def test_started_then_completed_event_order() -> None:
    bus = EventBus()
    registry = ToolRegistry()
    registry.register(SuccessfulTool())

    events: list[Event] = []

    bus.subscribe("tool.started", events.append)
    bus.subscribe("tool.completed", events.append)

    executor = ToolExecutor(registry, bus)

    await executor.execute("successful_tool")

    assert [e.name for e in events] == [
        "tool.started",
        "tool.completed",
    ]


@pytest.mark.anyio
async def test_started_then_failed_event_order() -> None:
    bus = EventBus()
    registry = ToolRegistry()
    registry.register(FailedTool())

    events: list[Event] = []

    bus.subscribe("tool.started", events.append)
    bus.subscribe("tool.failed", events.append)

    executor = ToolExecutor(registry, bus)

    await executor.execute("failed_tool")

    assert [e.name for e in events] == [
        "tool.started",
        "tool.failed",
    ]


def test_executor_creation() -> None:
    executor = ToolExecutor(
        ToolRegistry(),
        EventBus(),
    )

    assert executor is not None