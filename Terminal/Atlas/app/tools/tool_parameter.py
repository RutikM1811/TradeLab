from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ToolParameter:
    """Describes a single tool parameter."""

    name: str
    description: str
    type: str = "string"
    required: bool = True