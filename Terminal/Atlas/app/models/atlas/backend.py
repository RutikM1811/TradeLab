"""
Atlas inference backend contract.

Defines how Atlas communicates with any underlying inference engine.
"""

from abc import ABC, abstractmethod
from typing import Any

from app.types.model_result import ModelResult


class AbstractInferenceBackend(ABC):
    """Base contract for Atlas inference backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique backend name."""
        raise NotImplementedError

    @abstractmethod
    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        """Generate a response using the inference backend."""
        raise NotImplementedError