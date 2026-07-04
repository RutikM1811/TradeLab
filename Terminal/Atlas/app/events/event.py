"""
Core event definition for Atlas.

Every event published inside Atlas is represented by the Event class.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class Event:
    """Immutable event emitted by an Atlas component."""

    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )