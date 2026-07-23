from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class ToolCall:
    """Represents a request to execute a tool."""

    tool_name: str
    arguments: dict[str, Any]