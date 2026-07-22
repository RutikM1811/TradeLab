"""
Atlas AI Model.

The main Atlas model implementation. Atlas owns the model-facing
interface while inference is delegated to a replaceable backend.
"""

from typing import Any
from collections.abc import AsyncIterator
from app.contracts.abstract_model import AbstractModel
from app.memory.conversation import Conversation
from app.models.atlas.backend import AbstractInferenceBackend
from app.prompts.context_builder import ContextBuilder
from app.types.model_result import ModelResult


class AtlasModel(AbstractModel):
    """Atlas AI model backed by a replaceable inference engine."""

    SYSTEM_PROMPT = (
        "You are Atlas, an AI assistant. "
        "Be helpful, accurate, clear, and concise. "
        "Use the conversation history to maintain context across turns."
    )

    def __init__(
            self,
            backend: AbstractInferenceBackend,
            context_builder: ContextBuilder | None = None,
    ) -> None:
        self._backend = backend
        self._context_builder = (
                context_builder or ContextBuilder()
        )

    @property
    def name(self) -> str:
        """Return the model name."""

        return "atlas"

    @property
    def provider(self) -> str:
        """Return the model provider."""

        return "atlas"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        """Generate a response through the configured backend."""

        result = await self._backend.generate(
            prompt,
            **kwargs,
        )

        return self._with_metadata(result)
    async def generate_stream(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a response through the configured backend."""

        async for chunk in self._backend.generate_stream(
                prompt,
                **kwargs,
        ):
            yield chunk
    async def generate_from_conversation(
            self,
            conversation: Conversation,
            **kwargs: Any,
    ) -> ModelResult:
        """Generate a response from conversation history."""

        if len(conversation) == 0:
            return ModelResult.fail(
                error="Conversation cannot be empty."
            )

        structured_generate = getattr(
            self._backend,
            "generate_from_conversation",
            None,
        )

        if structured_generate is not None:
            try:
                result = await structured_generate(
                    conversation,
                    system_prompt=self.SYSTEM_PROMPT,
                    **kwargs,
                )
            except NotImplementedError:
                result = await self._generate_from_context(
                    conversation,
                    **kwargs,
                )
        else:
            result = await self._generate_from_context(
                conversation,
                **kwargs,
            )

        return self._with_metadata(result)

    async def _generate_from_context(
            self,
            conversation: Conversation,
            **kwargs: Any,
    ) -> ModelResult:
        """Generate using the flattened conversation context."""

        context = self._context_builder.build(
            conversation
        )

        return await self._backend.generate(
            context,
            **kwargs,
        )

    def _with_metadata(
            self,
            result: ModelResult,
    ) -> ModelResult:
        """Add Atlas metadata to a successful result."""

        if not result.success:
            return result

        metadata = {
            **result.metadata,
            "provider": self.provider,
            "model": self.name,
            "backend": self._backend.name,
        }

        return ModelResult.ok(
            content=result.content or "",
            metadata=metadata,
        )
    async def generate_stream_from_conversation(
            self,
            conversation: Conversation,
            **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a response from conversation history."""

        if len(conversation) == 0:
            raise ValueError(
                "Conversation cannot be empty."
            )

        structured_generate = getattr(
            self._backend,
            "generate_stream_from_conversation",
            None,
        )

        if structured_generate is not None:
            try:
                async for chunk in structured_generate(
                        conversation,
                        system_prompt=self.SYSTEM_PROMPT,
                        **kwargs,
                ):
                    yield chunk
                return

            except NotImplementedError:
                pass

        context = self._context_builder.build(
            conversation
        )

        async for chunk in self._backend.generate_stream(
                context,
                **kwargs,
        ):
            yield chunk