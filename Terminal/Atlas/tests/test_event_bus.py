from app.events.event import Event
from app.events.event_bus import EventBus


def test_event_bus_publishes_event_to_subscriber() -> None:
    event_bus = EventBus()
    received_events: list[Event] = []

    def handler(event: Event) -> None:
        received_events.append(event)

    event_bus.subscribe("kernel.started", handler)

    published_event = event_bus.publish(
        "kernel.started",
        {"version": "0.1.0"},
    )

    assert len(received_events) == 1
    assert received_events[0] == published_event
    assert published_event.name == "kernel.started"
    assert published_event.payload["version"] == "0.1.0"