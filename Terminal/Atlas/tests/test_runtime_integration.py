from typing import Any

import pytest

from app.contracts.abstract_model import AbstractModel
from app.contracts.abstract_tool import AbstractTool
from app.events.event import Event
from app.events.event_bus import EventBus
from app.kernel.bootstrap import Kernel
from app.models.model_manager import ModelManager
from app.models.model_registry import ModelRegistry
from app.tools.tool_executor import ToolExecutor
from app.tools.tool_registry import ToolRegistry
from app.types.model_result import ModelResult
from app.types.tool_result import ToolResult


class RuntimeTool(AbstractTool):
    @property
    def name(self) -> str:
        return "runtime_tool"

    @property
    def description(self) -> str:
        return "Tool used for runtime integration testing."

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult.ok(data=kwargs)


class RuntimeModel(AbstractModel):
    @property
    def name(self) -> str:
        return "runtime_model"

    @property
    def provider(self) -> str:
        return "atlas_test"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        return ModelResult.ok(
            content=f"Generated: {prompt}"
        )


@pytest.mark.anyio
async def test_kernel_executes_builtin_tool() -> None:
    kernel = Kernel()
    kernel.boot()

    result = await kernel.execute_tool("system_info")

    assert result.success is True
    assert result.data is not None


@pytest.mark.anyio
async def test_kernel_generates_with_builtin_model() -> None:
    kernel = Kernel()
    kernel.boot()

    result = await kernel.generate(
        "echo",
        "Hello Atlas",
    )

    assert result.success is True
    assert result.content == "Hello Atlas"


@pytest.mark.anyio
async def test_tool_runtime_passes_arguments() -> None:
    event_bus = EventBus()
    registry = ToolRegistry()
    registry.register(RuntimeTool())

    executor = ToolExecutor(registry, event_bus)

    result = await executor.execute(
        "runtime_tool",
        symbol="BTCUSDT",
        timeframe="1h",
    )

    assert result.data == {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
    }


@pytest.mark.anyio
async def test_model_runtime_passes_prompt() -> None:
    event_bus = EventBus()
    registry = ModelRegistry()
    registry.register(RuntimeModel())

    manager = ModelManager(registry, event_bus)

    result = await manager.generate(
        "runtime_model",
        "Analyze BTC",
    )

    assert result.content == "Generated: Analyze BTC"


@pytest.mark.anyio
async def test_tool_runtime_emits_events_in_order() -> None:
    event_bus = EventBus()
    registry = ToolRegistry()
    registry.register(RuntimeTool())

    events: list[Event] = []

    event_bus.subscribe("tool.started", events.append)
    event_bus.subscribe("tool.completed", events.append)

    executor = ToolExecutor(registry, event_bus)

    await executor.execute("runtime_tool")

    assert [event.name for event in events] == [
        "tool.started",
        "tool.completed",
    ]


@pytest.mark.anyio
async def test_model_runtime_emits_events_in_order() -> None:
    event_bus = EventBus()
    registry = ModelRegistry()
    registry.register(RuntimeModel())

    events: list[Event] = []

    event_bus.subscribe("model.started", events.append)
    event_bus.subscribe("model.completed", events.append)

    manager = ModelManager(registry, event_bus)

    await manager.generate(
        "runtime_model",
        "Hello",
    )

    assert [event.name for event in events] == [
        "model.started",
        "model.completed",
    ]


@pytest.mark.anyio
async def test_tool_started_event_contains_tool_name() -> None:
    event_bus = EventBus()
    registry = ToolRegistry()
    registry.register(RuntimeTool())

    events: list[Event] = []

    event_bus.subscribe("tool.started", events.append)

    executor = ToolExecutor(registry, event_bus)

    await executor.execute("runtime_tool")

    assert events[0].payload["tool_name"] == "runtime_tool"


@pytest.mark.anyio
async def test_model_started_event_contains_model_details() -> None:
    event_bus = EventBus()
    registry = ModelRegistry()
    registry.register(RuntimeModel())

    events: list[Event] = []

    event_bus.subscribe("model.started", events.append)

    manager = ModelManager(registry, event_bus)

    await manager.generate(
        "runtime_model",
        "Hello",
    )

    assert events[0].payload["model_name"] == "runtime_model"
    assert events[0].payload["provider"] == "atlas_test"


@pytest.mark.anyio
async def test_kernel_raises_for_unknown_tool() -> None:
    kernel = Kernel()
    kernel.boot()

    with pytest.raises(KeyError):
        await kernel.execute_tool("unknown_tool")


@pytest.mark.anyio
async def test_kernel_raises_for_unknown_model() -> None:
    kernel = Kernel()
    kernel.boot()

    with pytest.raises(KeyError):
        await kernel.generate(
            "unknown_model",
            "Hello",
        )