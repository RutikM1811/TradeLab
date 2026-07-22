"""
Atlas Chat Runtime.

Coordinates conversation state and Atlas model generation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from app.memory.conversation import Conversation
from app.memory.conversation_manager import ConversationManager
from app.models.atlas.atlas_model import AtlasModel
from app.services.tool_router import ToolRouter
from app.tools.tool_executor import ToolExecutor
from app.types.model_result import ModelResult


class ChatRuntime:

    def __init__(
            self,
            model: AtlasModel,
            conversation_manager: ConversationManager | None = None,
            tool_router: ToolRouter | None = None,
            tool_executor: ToolExecutor | None = None,
    ) -> None:
        self._model = model

        self._conversation_manager = (
            conversation_manager
            if conversation_manager is not None
            else ConversationManager()
        )

        self._tool_router = (
            tool_router
            if tool_router is not None
            else ToolRouter()
        )

        self._tool_executor = tool_executor

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
        """Send a message to Atlas."""

        conversation.add_user(message)

        tool_name = self._tool_router.route(message)

        if (
                tool_name is not None
                and self._tool_executor is not None
        ):
            tool_result = await self._tool_executor.execute(
                tool_name,
                **kwargs,
            )

            return ModelResult.ok(
                str(tool_result.data)
                if tool_result.success
                else tool_result.error or "",
            )

        # Generate using the model
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
                result.content,
            )

        return result

    async def send_stream(
            self,
            conversation: Conversation,
            message: str,
            **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Send a message and stream the assistant response."""

        conversation.add_user(message)

        self._tool_router.route(message)

        if (
                conversation.metadata.title == "New Conversation"
                and len(conversation) == 1
        ):
            conversation.metadata.rename(
                self._create_title(message)
            )

        chunks: list[str] = []

        async for chunk in self._model.generate_stream_from_conversation(
                conversation,
                **kwargs,
        ):
            chunks.append(chunk)
            yield chunk

        response = "".join(chunks)

        if response.strip():
            conversation.add_assistant(
                response,
            )

    async def send_to(
            self,
            conversation_id: UUID,
            message: str,
            **kwargs: Any,
    ) -> ModelResult:
        """Send a message to a managed conversation."""

        conversation = self._conversation_manager.get(
            conversation_id,
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
            message.strip().split(),
        )

        if len(title) <= max_length:
            return title

        return title[:max_length].rstrip() + "..."