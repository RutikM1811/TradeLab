"""
Atlas Event Bus.

Provides internal publish and subscribe communication between
framework components.
"""

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from app.events.event import Event


EventHandler = Callable[[Event], None]


class EventBus:
    """Publish and subscribe to events inside Atlas."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(
            self,
            event_name: str,
            handler: EventHandler,
    ) -> None:
        """Subscribe a handler to an event."""

        if handler not in self._subscribers[event_name]:
            self._subscribers[event_name].append(handler)

    def publish(
            self,
            event_name: str,
            payload: dict[str, Any] | None = None,
    ) -> Event:
        """Create and publish an event to all subscribed handlers."""

        event = Event(
            name=event_name,
            payload=payload or {},
        )

        for handler in self._subscribers[event_name]:
            handler(event)

        return event