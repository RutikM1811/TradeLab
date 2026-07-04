"""
Atlas Dependency Container.

Stores and resolves shared framework dependencies.
"""

from typing import Any, TypeVar

T = TypeVar("T")


class Container:
    """Stores shared Atlas dependencies."""

    def __init__(self) -> None:
        self._services: dict[type[Any], Any] = {}

    def register(self, service_type: type[T], instance: T) -> None:
        """Register a shared service instance."""

        if service_type in self._services:
            raise ValueError(
                f"{service_type.__name__} is already registered."
            )

        self._services[service_type] = instance

    def resolve(self, service_type: type[T]) -> T:
        """Resolve a registered service by its type."""

        if service_type not in self._services:
            raise KeyError(
                f"{service_type.__name__} is not registered."
            )

        return self._services[service_type]

    def contains(self, service_type: type[Any]) -> bool:
        """Return whether a service type is registered."""

        return service_type in self._services