"""
Atlas AI Model.

The main Atlas model implementation. Atlas owns the model-facing
interface while inference is delegated to a replaceable backend.
"""

from typing import Any

from app.contracts.abstract_model import AbstractModel
from app.memory.conversation import Conversation
from app.models.atlas.backend import AbstractInferenceBackend
from app.prompts.context_builder import ContextBuilder
from app.types.model_result import ModelResult


class AtlasModel(AbstractModel):
    """Atlas AI model backed by a replaceable inference engine."""

    def __init__(
            self,
            backend: AbstractInferenceBackend,
            context_builder: ContextBuilder | None = None,
    ) -> None:
        self._backend = backend
        self._context_builder = context_builder or ContextBuilder()

    @property
    def name(self) -> str:
        return "atlas"

    @property
    def provider(self) -> str:
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

    async def generate_from_conversation(
            self,
            conversation: Conversation,
            **kwargs: Any,
    ) -> ModelResult:
        """Generate a response from conversation history."""

        context = self._context_builder.build(conversation)

        if not context:
            return ModelResult.fail(
                error="Conversation cannot be empty."
            )

        return await self.generate(
            context,
            **kwargs,
        )