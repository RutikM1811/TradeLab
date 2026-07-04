"""
Base contract for all Atlas AI models.

Every model provider must implement this contract so the rest of
Atlas remains independent of OpenAI, Claude, Gemini, Ollama, or any
other model provider.
"""

from abc import ABC, abstractmethod
from typing import Any

from app.types.model_result import ModelResult


class AbstractModel(ABC):
    """Base contract that every Atlas AI model must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique model name."""
        raise NotImplementedError

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the model provider name."""
        raise NotImplementedError

    @abstractmethod
    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        """Generate a response from the model."""
        raise NotImplementedError