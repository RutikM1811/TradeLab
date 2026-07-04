from typing import Any

import pytest

from app.contracts.abstract_tool import AbstractTool
from app.tools.tool_registry import ToolRegistry
from app.types.tool_result import ToolResult


class DummyTool(AbstractTool):
    @property
    def name(self) -> str:
        return "dummy_tool"

    @property
    def description(self) -> str:
        return "A dummy tool used for testing."

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult.ok(data=kwargs)


class EmptyNameTool(DummyTool):
    @property
    def name(self) -> str:
        return "   "


def test_tool_registry_registers_and_gets_tool() -> None:
    registry = ToolRegistry()
    tool = DummyTool()

    registry.register(tool)

    assert registry.get("dummy_tool") is tool
    assert registry.contains("dummy_tool")


def test_tool_registry_rejects_duplicate_tool() -> None:
    registry = ToolRegistry()

    registry.register(DummyTool())

    with pytest.raises(ValueError):
        registry.register(DummyTool())


def test_tool_registry_rejects_empty_tool_name() -> None:
    registry = ToolRegistry()

    with pytest.raises(ValueError):
        registry.register(EmptyNameTool())


def test_tool_registry_raises_for_missing_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(KeyError):
        registry.get("missing_tool")


def test_tool_registry_returns_all_tools() -> None:
    registry = ToolRegistry()
    tool = DummyTool()

    registry.register(tool)

    tools = registry.all()

    assert tools == (tool,)
    assert isinstance(tools, tuple)