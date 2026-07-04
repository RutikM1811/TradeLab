"""
Atlas Tool Executor.

Executes registered tools and publishes tool lifecycle events.
"""

from typing import Any

from app.events.event_bus import EventBus
from app.tools.tool_registry import ToolRegistry
from app.types.tool_result import ToolResult


class ToolExecutor:
    """Executes Atlas tools through the shared tool registry."""

    def __init__(
            self,
            tool_registry: ToolRegistry,
            event_bus: EventBus,
    ) -> None:
        self._tool_registry = tool_registry
        self._event_bus = event_bus

    async def execute(
            self,
            tool_name: str,
            **kwargs: Any,
    ) -> ToolResult:
        """Execute a registered tool and publish lifecycle events."""

        tool = self._tool_registry.get(tool_name)

        self._event_bus.publish(
            "tool.started",
            {
                "tool_name": tool.name,
            },
        )

        try:
            result = await tool.execute(**kwargs)
        except Exception as exc:
            self._event_bus.publish(
                "tool.failed",
                {
                    "tool_name": tool.name,
                    "error": str(exc),
                },
            )

            return ToolResult.fail(error=str(exc))

        event_name = (
            "tool.completed"
            if result.success
            else "tool.failed"
        )

        self._event_bus.publish(
            event_name,
            {
                "tool_name": tool.name,
                "success": result.success,
            },
        )

        return result