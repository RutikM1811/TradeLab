from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
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

    async def generate_stream(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream a response incrementally.

        Default implementation wraps generate().
        Backends may override this with native streaming.
        """

        result = await self.generate(
            prompt,
            **kwargs,
        )

        if result.success and result.content:
            yield result.content

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

    async def generate_stream_from_conversation(
            self,
            conversation: Conversation,
            system_prompt: str | None = None,
            **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream a response from a structured conversation.

        Default implementation wraps
        generate_from_conversation().
        Backends may override this.
        """

        result = await self.generate_from_conversation(
            conversation,
            system_prompt=system_prompt,
            **kwargs,
        )

        if result.success and result.content:
            yield result.content