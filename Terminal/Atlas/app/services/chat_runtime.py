"""
Atlas Chat Runtime.

Coordinates conversation state and Atlas model generation.
"""

from typing import Any
from uuid import UUID

from app.memory.conversation import Conversation
from app.memory.conversation_manager import ConversationManager
from app.models.atlas.atlas_model import AtlasModel
from app.types.model_result import ModelResult


class ChatRuntime:
    """Coordinates multi-turn conversations with Atlas."""

    def __init__(
            self,
            model: AtlasModel,
            conversation_manager: ConversationManager | None = None,
    ) -> None:
        self._model = model
        self._conversation_manager = (
            conversation_manager
            if conversation_manager is not None
            else ConversationManager()
        )

    def create_conversation(
            self,
    ) -> tuple[UUID, Conversation]:
        """Create and return a managed conversation."""

        return self._conversation_manager.create()

    async def send(
            self,
            conversation: Conversation,
            message: str,
            **kwargs: Any,
    ) -> ModelResult:
        """Send a message using a conversation instance."""

        conversation.add_user(message)

        result = await self._model.generate_from_conversation(
            conversation,
            **kwargs,
        )

        if (
                conversation.metadata.title == "New Conversation"
                and len(conversation) == 1
        ):
            conversation.metadata.rename(
                self._create_title(message)
            )

        if (
                result.success
                and result.content is not None
                and result.content.strip()
        ):
            conversation.add_assistant(
                result.content
            )

        return result

    async def send_to(
            self,
            conversation_id: UUID,
            message: str,
            **kwargs: Any,
    ) -> ModelResult:
        """Send a message to a managed conversation."""

        conversation = self._conversation_manager.get(
            conversation_id
        )

        return await self.send(
            conversation,
            message,
            **kwargs,
        )

    @staticmethod
    def _create_title(
            message: str,
            max_length: int = 50,
    ) -> str:
        """Create a conversation title from the first user message."""

        title = " ".join(
            message.strip().split()
        )

        if len(title) <= max_length:
            return title

        return title[:max_length].rstrip() + "..."