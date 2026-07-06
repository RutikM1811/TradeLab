"""
Atlas Conversation Serializer.

Converts conversations to and from JSON-compatible data.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from app.memory.conversation import Conversation
from app.memory.conversation_metadata import ConversationMetadata
from app.types.message import Message, MessageRole


class ConversationSerializer:
    """Serialize and restore Atlas conversations."""

    def serialize(
            self,
            conversation_id: UUID,
            conversation: Conversation,
    ) -> dict[str, Any]:
        """Convert a conversation into JSON-compatible data."""

        return {
            "id": str(conversation_id),
            "metadata": {
                "title": conversation.metadata.title,
                "created_at": (
                    conversation.metadata.created_at.isoformat()
                ),
                "updated_at": (
                    conversation.metadata.updated_at.isoformat()
                ),
            },
            "messages": [
                {
                    "id": str(message.id),
                    "role": message.role.value,
                    "content": message.content,
                    "created_at": (
                        message.created_at.isoformat()
                    ),
                    "metadata": message.metadata,
                }
                for message in conversation.all()
            ],
        }

    def deserialize(
            self,
            data: dict[str, Any],
    ) -> tuple[UUID, Conversation]:
        """Restore a conversation from serialized data."""

        conversation_id = UUID(data["id"])

        metadata_data = data.get("metadata")

        if metadata_data is None:
            metadata = ConversationMetadata()
        else:
            metadata = ConversationMetadata(
                title=metadata_data["title"],
                created_at=datetime.fromisoformat(
                    metadata_data["created_at"]
                ),
                updated_at=datetime.fromisoformat(
                    metadata_data["updated_at"]
                ),
            )

        conversation = Conversation(
            metadata=metadata,
        )

        for message_data in data.get("messages", []):
            message = Message(
                id=UUID(message_data["id"]),
                role=MessageRole(message_data["role"]),
                content=message_data["content"],
                created_at=datetime.fromisoformat(
                    message_data["created_at"]
                ),
                metadata=message_data.get(
                    "metadata",
                    {},
                ),
            )

            conversation.add(message)

        return conversation_id, conversation