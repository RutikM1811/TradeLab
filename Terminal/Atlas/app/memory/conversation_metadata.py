"""
Atlas Conversation Metadata.

Stores descriptive information about a conversation.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(slots=True)
class ConversationMetadata:
    """Metadata associated with an Atlas conversation."""

    title: str = "New Conversation"

    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    updated_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    def rename(
            self,
            title: str,
    ) -> None:
        """Rename the conversation."""

        cleaned_title = title.strip()

        if not cleaned_title:
            raise ValueError(
                "Conversation title cannot be empty."
            )

        self.title = cleaned_title
        self.touch()

    def touch(self) -> None:
        """Update the conversation modification time."""

        self.updated_at = datetime.now(UTC)