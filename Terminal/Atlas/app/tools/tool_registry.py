"""
Atlas Tool Registry.

Registers and resolves tools available to Atlas.
"""

from app.contracts.abstract_tool import AbstractTool


class ToolRegistry:
    """Stores and discovers Atlas tools."""

    def __init__(self) -> None:
        self._tools: dict[str, AbstractTool] = {}

    def register(self, tool: AbstractTool) -> None:
        """Register a tool by its unique name."""

        name = tool.name.strip()

        if not name:
            raise ValueError("Tool name cannot be empty.")

        if name in self._tools:
            raise ValueError(
                f"Tool '{name}' is already registered."
            )

        self._tools[name] = tool

    def get(self, name: str) -> AbstractTool:
        """Return a registered tool by name."""

        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' is not registered."
            )

        return self._tools[name]

    def contains(self, name: str) -> bool:
        """Return whether a tool is registered."""

        return name in self._tools

    def all(self) -> tuple[AbstractTool, ...]:
        """Return all registered tools."""

        return tuple(self._tools.values())