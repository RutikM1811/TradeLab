import pytest

from app.events.event import Event
from app.events.event_bus import EventBus
from app.models.echo_model import EchoModel
from app.models.model_manager import ModelManager
from app.models.model_registry import ModelRegistry


@pytest.mark.anyio
async def test_echo_model_executes_through_pipeline() -> None:
    event_bus = EventBus()
    model_registry = ModelRegistry()

    model_registry.register(EchoModel())

    manager = ModelManager(
        model_registry=model_registry,
        event_bus=event_bus,
    )

    result = await manager.generate(
        "echo",
        "Hello Atlas",
    )

    assert result.success is True
    assert result.content == "Hello Atlas"
    assert result.error is None
    assert result.metadata["provider"] == "atlas"
    assert result.metadata["model"] == "echo"


@pytest.mark.anyio
async def test_echo_model_publishes_lifecycle_events() -> None:
    event_bus = EventBus()
    model_registry = ModelRegistry()

    model_registry.register(EchoModel())

    received_events: list[Event] = []

    event_bus.subscribe(
        "model.started",
        received_events.append,
    )
    event_bus.subscribe(
        "model.completed",
        received_events.append,
    )

    manager = ModelManager(
        model_registry=model_registry,
        event_bus=event_bus,
    )

    await manager.generate(
        "echo",
        "Hello Atlas",
    )

    assert [event.name for event in received_events] == [
        "model.started",
        "model.completed",
    ]