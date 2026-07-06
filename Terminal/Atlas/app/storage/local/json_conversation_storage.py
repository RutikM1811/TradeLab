"""
Atlas Local JSON Conversation Storage.

Persists serialized conversations as JSON files.
"""

import json
from pathlib import Path
from uuid import UUID

from app.memory.conversation import Conversation
from app.memory.conversation_serializer import ConversationSerializer


class JsonConversationStorage:
    """Store Atlas conversations as local JSON files."""

    def __init__(
            self,
            directory: str | Path,
            serializer: ConversationSerializer | None = None,
    ) -> None:
        self._directory = Path(directory)
        self._serializer = serializer or ConversationSerializer()

        self._directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
            self,
            conversation_id: UUID,
            conversation: Conversation,
    ) -> None:
        """Save or overwrite a conversation."""

        data = self._serializer.serialize(
            conversation_id,
            conversation,
        )

        path = self._path_for(conversation_id)

        path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(
            self,
            conversation_id: UUID,
    ) -> Conversation:
        """Load a conversation by ID."""

        path = self._path_for(conversation_id)

        if not path.exists():
            raise KeyError(
                f"Conversation '{conversation_id}' was not found."
            )

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        restored_id, conversation = (
            self._serializer.deserialize(data)
        )

        if restored_id != conversation_id:
            raise ValueError(
                "Stored conversation ID does not match requested ID."
            )

        return conversation

    def contains(
            self,
            conversation_id: UUID,
    ) -> bool:
        """Return whether a stored conversation exists."""

        return self._path_for(conversation_id).exists()

    def all_ids(self) -> tuple[UUID, ...]:
        """Return all persisted conversation IDs."""

        conversation_ids: list[UUID] = []

        for path in sorted(self._directory.glob("*.json")):
            conversation_ids.append(
                UUID(path.stem)
            )

        return tuple(conversation_ids)

    def delete(
            self,
            conversation_id: UUID,
    ) -> None:
        """Delete a persisted conversation."""

        path = self._path_for(conversation_id)

        if not path.exists():
            raise KeyError(
                f"Conversation '{conversation_id}' was not found."
            )

        path.unlink()

    def clear(self) -> None:
        """Delete all persisted conversations."""

        for path in self._directory.glob("*.json"):
            path.unlink()

    def _path_for(
            self,
            conversation_id: UUID,
    ) -> Path:
        return self._directory / f"{conversation_id}.json"