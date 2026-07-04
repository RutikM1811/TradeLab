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