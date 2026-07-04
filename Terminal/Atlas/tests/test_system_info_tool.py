import pytest

from app.events.event_bus import EventBus
from app.tools.system_info_tool import SystemInfoTool
from app.tools.tool_executor import ToolExecutor
from app.tools.tool_registry import ToolRegistry


@pytest.mark.anyio
async def test_system_info_tool_executes_through_pipeline() -> None:
    event_bus = EventBus()
    tool_registry = ToolRegistry()

    tool_registry.register(SystemInfoTool())

    executor = ToolExecutor(
        tool_registry=tool_registry,
        event_bus=event_bus,
    )

    result = await executor.execute("system_info")

    assert result.success is True
    assert result.error is None
    assert result.data is not None

    assert "python_version" in result.data
    assert "operating_system" in result.data
    assert "platform" in result.data