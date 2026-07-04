"""
Atlas system information tool.

Provides basic runtime information about the Atlas environment.
"""

import platform
import sys
from typing import Any

from app.contracts.abstract_tool import AbstractTool
from app.types.tool_result import ToolResult


class SystemInfoTool(AbstractTool):
    """Return information about the current Atlas runtime."""

    @property
    def name(self) -> str:
        return "system_info"

    @property
    def description(self) -> str:
        return "Returns Python, operating system, and platform information."

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Collect and return runtime information."""

        return ToolResult.ok(
            data={
                "python_version": sys.version,
                "operating_system": platform.system(),
                "platform": platform.platform(),
            }
        )