"""
Atlas Chat Runtime.

Coordinates conversation state and Atlas model generation.
"""

from typing import Any

from app.memory.conversation import Conversation
from app.models.atlas.atlas_model import AtlasModel
from app.types.model_result import ModelResult


class ChatRuntime:
    """Coordinates multi-turn conversations with Atlas."""

    def __init__(
            self,
            model: AtlasModel,
    ) -> None:
        self._model = model

    async def send(
            self,
            conversation: Conversation,
            message: str,
            **kwargs: Any,
    ) -> ModelResult:
        """Send a user message and store the generated response."""

        conversation.add_user(message)

        result = await self._model.generate_from_conversation(
            conversation,
            **kwargs,
        )

        if result.success and result.content is not None:
            conversation.add_assistant(result.content)

        return result