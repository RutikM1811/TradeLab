"""
Atlas Model Registry.

Registers and resolves AI models available to Atlas.
"""

from app.contracts.abstract_model import AbstractModel

class ModelRegistry:
    """Stores and discovers Atlas AI models."""

    def __init__(self) -> None:
        self._models: dict[str, AbstractModel] = {}

    def register(self, model: AbstractModel) -> None:
        """Register a model by its unique name."""

        name = model.name.strip()

        if not name:
            raise ValueError("Model name cannot be empty.")

        if name in self._models:
            raise ValueError(
                f"Model '{name}' is already registered."
            )

        self._models[name] = model

    def get(self, name: str) -> AbstractModel:
        """Return a registered model by name."""

        if name not in self._models:
            raise KeyError(
                f"Model '{name}' is not registered."
            )

        return self._models[name]

    def contains(self, name: str) -> bool:
        """Return whether a model is registered."""

        return name in self._models

    def all(self) -> tuple[AbstractModel, ...]:
        """Return all registered models."""

        return tuple(self._models.values())