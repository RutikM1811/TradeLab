"""
Atlas conversation message types.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4


class MessageRole(StrEnum):
    """Roles supported by the Atlas conversation system."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class Message:
    """Represents one message in an Atlas conversation."""

    role: MessageRole
    content: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )
    metadata: dict[str, Any] = field(default_factory=dict)