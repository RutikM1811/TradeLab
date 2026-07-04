"""
Standard result returned by Atlas tools.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Represents the result of a tool execution."""

    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
            cls,
            data: Any = None,
            metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        """Create a successful tool result."""

        return cls(
            success=True,
            data=data,
            metadata=metadata or {},
        )

    @classmethod
    def fail(
            cls,
            error: str,
            metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        """Create a failed tool result."""

        return cls(
            success=False,
            error=error,
            metadata=metadata or {},
        )