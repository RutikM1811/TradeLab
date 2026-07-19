from abc import ABC, abstractmethod
from typing import Any

from app.memory.conversation import Conversation
from app.types.model_result import ModelResult


class AbstractInferenceBackend(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name."""
        raise NotImplementedError

    @abstractmethod
    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        """Generate a response."""
        raise NotImplementedError

    async def generate_from_conversation(
            self,
            conversation: Conversation,
            system_prompt: str | None = None,
            **kwargs: Any,
    ) -> ModelResult:
        """
        Default implementation builds a prompt from the conversation.
        Backends may override this.
        """
        raise NotImplementedError(
            f"{self.name} backend does not support "
            "structured conversations."
        )