"""
Atlas Terminal Command Runtime.

Handles commands for interactive conversation management.
"""

from typing import Any
from uuid import UUID

from app.kernel.bootstrap import Kernel
from app.memory.conversation import Conversation


class TerminalCommandRuntime:
    """Handles Atlas terminal commands."""

    def __init__(
            self,
            kernel: Kernel,
    ) -> None:
        self._kernel = kernel

    def create_conversation(
            self,
    ) -> tuple[UUID, Conversation]:
        """Create a new active conversation."""

        return self._kernel.create_conversation()

    def rename_conversation(
            self,
            conversation_id: UUID,
            title: str,
    ) -> None:
        """Rename and persist a conversation."""

        conversation = self._kernel.get_conversation(
            conversation_id
        )

        conversation.metadata.rename(title)

        self._kernel.save_conversation(
            conversation_id
        )

    def list_conversations(
            self,
    ) -> tuple[tuple[UUID, Conversation], ...]:
        """Return all available conversations."""

        return self._kernel.list_conversations()

    def switch_conversation(
            self,
            index: int,
    ) -> tuple[UUID, Conversation]:
        """Return a conversation selected by one-based index."""

        conversations = self.list_conversations()

        if index < 1 or index > len(conversations):
            raise IndexError(
                f"Conversation number {index} does not exist."
            )

        return conversations[index - 1]

    def history(
            self,
            conversation_id: UUID,
    ) -> tuple[str, ...]:
        """Return formatted conversation history."""

        conversation = self._kernel.get_conversation(
            conversation_id
        )

        return tuple(
            f"{message.role.value.capitalize()}: "
            f"{message.content}"
            for message in conversation.all()
        )

    def conversation_info(
            self,
            conversation_id: UUID,
    ) -> dict[str, Any]:
        """Return information about a conversation."""

        conversation = self._kernel.get_conversation(
            conversation_id
        )

        return {
            "id": conversation_id,
            "title": conversation.metadata.title,
            "created_at": conversation.metadata.created_at,
            "updated_at": conversation.metadata.updated_at,
            "message_count": len(conversation),
        }

    def delete_conversation(
            self,
            conversation_id: UUID,
    ) -> None:
        """Delete a conversation."""

        self._kernel.delete_conversation(
            conversation_id
        )

    def help_text(self) -> tuple[str, ...]:
        """Return available terminal commands."""

        return (
            "/new - Create a new conversation",
            "/list - Show all conversations",
            "/switch <number> - Switch conversation",
            "/history - Show current conversation history",
            "/info - Show current conversation information",
            "/rename <title> - Rename current conversation",
            "/delete - Delete current conversation",
            "/help - Show available commands",
            "/exit - Save and stop Atlas",
        )