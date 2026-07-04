import asyncio
from typing import Any

import pytest

from app.contracts.abstract_tool import AbstractTool


class ValidTool(AbstractTool):
    @property
    def name(self) -> str:
        return "valid_tool"

    @property
    def description(self) -> str:
        return "A tool used for testing."

    async def execute(self, **kwargs: Any) -> Any:
        return kwargs


class IncompleteTool(AbstractTool):
    pass


def test_incomplete_tool_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        IncompleteTool()


def test_valid_tool_properties() -> None:
    tool = ValidTool()

    assert tool.name == "valid_tool"
    assert tool.description == "A tool used for testing."


def test_valid_tool_executes() -> None:
    tool = ValidTool()

    result = asyncio.run(
        tool.execute(symbol="BTCUSDT")
    )

    assert result == {"symbol": "BTCUSDT"}