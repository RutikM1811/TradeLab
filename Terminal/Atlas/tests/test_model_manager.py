from typing import Any

import pytest

from app.contracts.abstract_model import AbstractModel
from app.events.event import Event
from app.events.event_bus import EventBus
from app.models.model_manager import ModelManager
from app.models.model_registry import ModelRegistry
from app.types.model_result import ModelResult


class SuccessfulModel(AbstractModel):
    @property
    def name(self) -> str:
        return "successful_model"

    @property
    def provider(self) -> str:
        return "test_provider"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        return ModelResult.ok(
            content=f"Response to: {prompt}"
        )


class FailedModel(AbstractModel):
    @property
    def name(self) -> str:
        return "failed_model"

    @property
    def provider(self) -> str:
        return "test_provider"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        return ModelResult.fail(
            error="Model generation failed."
        )


class ExceptionModel(AbstractModel):
    @property
    def name(self) -> str:
        return "exception_model"

    @property
    def provider(self) -> str:
        return "test_provider"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        raise RuntimeError("Unexpected model error.")


@pytest.mark.anyio
async def test_model_manager_generates_successfully() -> None:
    event_bus = EventBus()
    model_registry = ModelRegistry()
    model_registry.register(SuccessfulModel())

    manager = ModelManager(
        model_registry=model_registry,
        event_bus=event_bus,
    )

    result = await manager.generate(
        "successful_model",
        "Analyze BTC",
    )

    assert result.success is True
    assert result.content == "Response to: Analyze BTC"


@pytest.mark.anyio
async def test_model_manager_handles_failed_result() -> None:
    event_bus = EventBus()
    model_registry = ModelRegistry()
    model_registry.register(FailedModel())

    manager = ModelManager(
        model_registry=model_registry,
        event_bus=event_bus,
    )

    result = await manager.generate(
        "failed_model",
        "Analyze BTC",
    )

    assert result.success is False
    assert result.error == "Model generation failed."


@pytest.mark.anyio
async def test_model_manager_handles_exception() -> None:
    event_bus = EventBus()
    model_registry = ModelRegistry()
    model_registry.register(ExceptionModel())

    manager = ModelManager(
        model_registry=model_registry,
        event_bus=event_bus,
    )

    result = await manager.generate(
        "exception_model",
        "Analyze BTC",
    )

    assert result.success is False
    assert result.error == "Unexpected model error."


@pytest.mark.anyio
async def test_model_manager_publishes_lifecycle_events() -> None:
    event_bus = EventBus()
    model_registry = ModelRegistry()
    model_registry.register(SuccessfulModel())

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
        "successful_model",
        "Analyze BTC",
    )

    assert [event.name for event in received_events] == [
        "model.started",
        "model.completed",
    ]


class KwargsModel(AbstractModel):
    def __init__(self) -> None:
        self.received_prompt: str | None = None
        self.received_kwargs: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "kwargs_model"

    @property
    def provider(self) -> str:
        return "kwargs_provider"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        self.received_prompt = prompt
        self.received_kwargs = kwargs
        return ModelResult.ok(content="done")


@pytest.mark.anyio
async def test_generate_empty_prompt() -> None:
    bus = EventBus()
    registry = ModelRegistry()
    registry.register(SuccessfulModel())

    manager = ModelManager(registry, bus)

    result = await manager.generate(
        "successful_model",
        "",
    )

    assert result.success


@pytest.mark.anyio
async def test_generate_unicode_prompt() -> None:
    bus = EventBus()
    registry = ModelRegistry()
    registry.register(SuccessfulModel())

    manager = ModelManager(registry, bus)

    result = await manager.generate(
        "successful_model",
        "नमस्कार 🚀",
    )

    assert result.success


@pytest.mark.anyio
async def test_generate_long_prompt() -> None:
    bus = EventBus()
    registry = ModelRegistry()
    registry.register(SuccessfulModel())

    manager = ModelManager(registry, bus)

    result = await manager.generate(
        "successful_model",
        "A" * 5000,
        )

    assert result.success


@pytest.mark.anyio
async def test_kwargs_are_forwarded() -> None:
    bus = EventBus()
    registry = ModelRegistry()

    model = KwargsModel()
    registry.register(model)

    manager = ModelManager(registry, bus)

    await manager.generate(
        "kwargs_model",
        "BTC",
        temperature=0.4,
        max_tokens=100,
    )

    assert model.received_prompt == "BTC"
    assert model.received_kwargs == {
        "temperature": 0.4,
        "max_tokens": 100,
    }


@pytest.mark.anyio
async def test_started_event_contains_model_name() -> None:
    bus = EventBus()
    registry = ModelRegistry()
    registry.register(SuccessfulModel())

    events = []

    bus.subscribe(
        "model.started",
        events.append,
    )

    manager = ModelManager(registry, bus)

    await manager.generate(
        "successful_model",
        "Hello",
    )

    assert events[0].payload["model_name"] == "successful_model"


@pytest.mark.anyio
async def test_started_event_contains_provider() -> None:
    bus = EventBus()
    registry = ModelRegistry()
    registry.register(SuccessfulModel())

    events = []

    bus.subscribe(
        "model.started",
        events.append,
    )

    manager = ModelManager(registry, bus)

    await manager.generate(
        "successful_model",
        "Hello",
    )

    assert events[0].payload["provider"] == "test_provider"


@pytest.mark.anyio
async def test_completed_event_contains_success() -> None:
    bus = EventBus()
    registry = ModelRegistry()
    registry.register(SuccessfulModel())

    events = []

    bus.subscribe(
        "model.completed",
        events.append,
    )

    manager = ModelManager(registry, bus)

    await manager.generate(
        "successful_model",
        "Hello",
    )

    assert events[0].payload["success"] is True


@pytest.mark.anyio
async def test_completed_event_contains_model_name() -> None:
    bus = EventBus()
    registry = ModelRegistry()
    registry.register(SuccessfulModel())

    events = []

    bus.subscribe(
        "model.completed",
        events.append,
    )

    manager = ModelManager(registry, bus)

    await manager.generate(
        "successful_model",
        "Hello",
    )

    assert events[0].payload["model_name"] == "successful_model"


@pytest.mark.anyio
async def test_completed_event_contains_provider() -> None:
    bus = EventBus()
    registry = ModelRegistry()
    registry.register(SuccessfulModel())

    events = []

    bus.subscribe(
        "model.completed",
        events.append,
    )

    manager = ModelManager(registry, bus)

    await manager.generate(
        "successful_model",
        "Hello",
    )

    assert events[0].payload["provider"] == "test_provider"


@pytest.mark.anyio
async def test_failed_result_publishes_failed_event() -> None:
    bus = EventBus()
    registry = ModelRegistry()
    registry.register(FailedModel())

    events = []

    bus.subscribe(
        "model.failed",
        events.append,
    )

    manager = ModelManager(registry, bus)

    await manager.generate(
        "failed_model",
        "Hello",
    )

    assert len(events) == 1


@pytest.mark.anyio
async def test_failed_event_contains_success_false() -> None:
    bus = EventBus()
    registry = ModelRegistry()
    registry.register(FailedModel())

    events = []

    bus.subscribe(
        "model.failed",
        events.append,
    )

    manager = ModelManager(registry, bus)

    await manager.generate(
        "failed_model",
        "Hello",
    )

    assert events[0].payload["success"] is False


@pytest.mark.anyio
async def test_failed_event_contains_model_name() -> None:
    bus = EventBus()
    registry = ModelRegistry()
    registry.register(FailedModel())

    events = []

    bus.subscribe(
        "model.failed",
        events.append,
    )

    manager = ModelManager(registry, bus)

    await manager.generate(
        "failed_model",
        "Hello",
    )

    assert events[0].payload["model_name"] == "failed_model"


@pytest.mark.anyio
async def test_failed_event_contains_provider() -> None:
    bus = EventBus()
    registry = ModelRegistry()
    registry.register(FailedModel())

    events = []

    bus.subscribe(
        "model.failed",
        events.append,
    )

    manager = ModelManager(registry, bus)

    await manager.generate(
        "failed_model",
        "Hello",
    )

    assert events[0].payload["provider"] == "test_provider"


@pytest.mark.anyio
async def test_exception_event_contains_error() -> None:
    bus = EventBus()
    registry = ModelRegistry()
    registry.register(ExceptionModel())

    events = []

    bus.subscribe(
        "model.failed",
        events.append,
    )

    manager = ModelManager(registry, bus)

    await manager.generate(
        "exception_model",
        "Hello",
    )

    assert events[0].payload["error"] == "Unexpected model error."


@pytest.mark.anyio
async def test_exception_event_contains_model_name() -> None:
    bus = EventBus()
    registry = ModelRegistry()
    registry.register(ExceptionModel())

    events = []

    bus.subscribe(
        "model.failed",
        events.append,
    )

    manager = ModelManager(registry, bus)

    await manager.generate(
        "exception_model",
        "Hello",
    )

    assert events[0].payload["model_name"] == "exception_model"


@pytest.mark.anyio
async def test_exception_event_contains_provider() -> None:
    bus = EventBus()
    registry = ModelRegistry()
    registry.register(ExceptionModel())

    events = []

    bus.subscribe(
        "model.failed",
        events.append,
    )

    manager = ModelManager(registry, bus)

    await manager.generate(
        "exception_model",
        "Hello",
    )

    assert events[0].payload["provider"] == "test_provider"


@pytest.mark.anyio
async def test_multiple_successful_generations() -> None:
    bus = EventBus()
    registry = ModelRegistry()
    registry.register(SuccessfulModel())

    manager = ModelManager(registry, bus)

    for _ in range(5):
        result = await manager.generate(
            "successful_model",
            "BTC",
        )
        assert result.success


@pytest.mark.anyio
async def test_multiple_failed_generations() -> None:
    bus = EventBus()
    registry = ModelRegistry()
    registry.register(FailedModel())

    manager = ModelManager(registry, bus)

    for _ in range(3):
        result = await manager.generate(
            "failed_model",
            "BTC",
        )
        assert not result.success


@pytest.mark.anyio
async def test_result_content_preserved() -> None:
    bus = EventBus()
    registry = ModelRegistry()
    registry.register(SuccessfulModel())

    manager = ModelManager(registry, bus)

    result = await manager.generate(
        "successful_model",
        "ABC",
    )

    assert result.content == "Response to: ABC"


@pytest.mark.anyio
async def test_result_error_preserved() -> None:
    bus = EventBus()
    registry = ModelRegistry()
    registry.register(FailedModel())

    manager = ModelManager(registry, bus)

    result = await manager.generate(
        "failed_model",
        "ABC",
    )

    assert result.error == "Model generation failed."


def test_manager_creation() -> None:
    manager = ModelManager(
        ModelRegistry(),
        EventBus(),
    )

    assert manager is not None