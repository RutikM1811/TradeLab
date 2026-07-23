from __future__ import annotations

from dataclasses import dataclass

from app.tools.tool_parameter import ToolParameter


@dataclass(slots=True, frozen=True)
class ToolSchema:
    """Describes a callable tool."""

    name: str
    description: str
    parameters: tuple[ToolParameter, ...] = ()
    returns: str = ""