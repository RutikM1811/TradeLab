"""
Atlas inference backend contract.

Defines how Atlas communicates with any underlying inference engine.
"""

from abc import ABC, abstractmethod
from typing import Any

from app.memory.conversation import Conversation
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

async def generate_from_conversation(
        self,
        conversation: Conversation,
        system_prompt: str | None = None,
        **kwargs: Any,
) -> ModelResult:
    """Generate from structured conversation history."""

    raise NotImplementedError(
        f"{self.name} backend does not support "
        "structured conversations."
    )