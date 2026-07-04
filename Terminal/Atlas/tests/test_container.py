from app.events.event_bus import EventBus
from app.kernel.container import Container


def test_container_registers_and_resolves_service() -> None:
    container = Container()
    event_bus = EventBus()

    container.register(EventBus, event_bus)

    resolved_event_bus = container.resolve(EventBus)

    assert resolved_event_bus is event_bus
    assert container.contains(EventBus)


def test_container_rejects_duplicate_registration() -> None:
    container = Container()

    container.register(EventBus, EventBus())

    try:
        container.register(EventBus, EventBus())
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_container_raises_for_missing_service() -> None:
    container = Container()

    try:
        container.resolve(EventBus)
        assert False, "Expected KeyError"
    except KeyError:
        pass