"""
Atlas Conversation Manager.

Creates, stores, resolves, lists, deletes, and persists
conversation sessions.
"""

from uuid import UUID, uuid4

from app.memory.conversation import Conversation
from app.storage.local.json_conversation_storage import (
    JsonConversationStorage,
)


class ConversationManager:
    """Manages active Atlas conversation sessions."""

    def __init__(
            self,
            storage: JsonConversationStorage | None = None,
    ) -> None:
        self._conversations: dict[UUID, Conversation] = {}
        self._storage = storage

    def create(self) -> tuple[UUID, Conversation]:
        """Create and store a new conversation."""

        conversation_id = uuid4()
        conversation = Conversation()

        self._conversations[conversation_id] = conversation

        return conversation_id, conversation

    def get(
            self,
            conversation_id: UUID,
    ) -> Conversation:
        """Return a conversation by its ID."""

        if conversation_id not in self._conversations:
            raise KeyError(
                f"Conversation '{conversation_id}' was not found."
            )

        return self._conversations[conversation_id]

    def contains(
            self,
            conversation_id: UUID,
    ) -> bool:
        """Return whether a conversation exists."""

        return conversation_id in self._conversations

    def all(
            self,
    ) -> tuple[tuple[UUID, Conversation], ...]:
        """Return all conversations in creation order."""

        return tuple(self._conversations.items())

    def delete(
            self,
            conversation_id: UUID,
    ) -> None:
        """Delete an existing conversation."""

        if conversation_id not in self._conversations:
            raise KeyError(
                f"Conversation '{conversation_id}' was not found."
            )

        del self._conversations[conversation_id]

        if (
                self._storage is not None
                and self._storage.contains(conversation_id)
        ):
            self._storage.delete(conversation_id)

    def clear(self) -> None:
        """Delete all conversations from memory."""

        self._conversations.clear()

    def save(
            self,
            conversation_id: UUID,
    ) -> None:
        """Persist a managed conversation."""

        if self._storage is None:
            raise RuntimeError(
                "Conversation storage is not configured."
            )

        conversation = self.get(conversation_id)

        self._storage.save(
            conversation_id,
            conversation,
        )

    def load(
            self,
            conversation_id: UUID,
    ) -> Conversation:
        """Load a persisted conversation into memory."""

        if self._storage is None:
            raise RuntimeError(
                "Conversation storage is not configured."
            )

        conversation = self._storage.load(
            conversation_id
        )

        self._conversations[
            conversation_id
        ] = conversation

        return conversation

    def restore_all(self) -> int:
        """Restore all persisted conversations into memory."""

        if self._storage is None:
            raise RuntimeError(
                "Conversation storage is not configured."
            )

        restored_count = 0

        for conversation_id in self._storage.all_ids():
            self.load(conversation_id)
            restored_count += 1

        return restored_count

    def __len__(self) -> int:
        """Return the number of stored conversations."""

        return len(self._conversations)