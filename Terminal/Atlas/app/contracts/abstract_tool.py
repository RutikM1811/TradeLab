"""
Base contract for all Atlas tools.

A tool is an atomic capability that can be executed by Atlas.
"""

from abc import ABC, abstractmethod
from typing import Any

from app.types.tool_result import ToolResult


class AbstractTool(ABC):
    """Base contract that every Atlas tool must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique tool name."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of the tool."""
        raise NotImplementedError

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool and return a standardized result."""
        raise NotImplementedError