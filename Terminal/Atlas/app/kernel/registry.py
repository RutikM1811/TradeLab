"""
Atlas Component Registry.

Stores and resolves named framework components.
"""

from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    """Stores named Atlas components."""

    def __init__(self) -> None:
        self._components: dict[str, Any] = {}

    def register(self, name: str, component: T) -> None:
        """Register a component under a unique name."""

        if not name.strip():
            raise ValueError("Component name cannot be empty.")

        if name in self._components:
            raise ValueError(
                f"Component '{name}' is already registered."
            )

        self._components[name] = component

    def get(self, name: str) -> Any:
        """Return a registered component by name."""

        if name not in self._components:
            raise KeyError(
                f"Component '{name}' is not registered."
            )

        return self._components[name]

    def contains(self, name: str) -> bool:
        """Return whether a component is registered."""

        return name in self._components

    def all(self) -> dict[str, Any]:
        """Return a copy of all registered components."""

        return self._components.copy()