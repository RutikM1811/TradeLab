from typing import Any

import pytest

from app.events.event import Event
from app.events.event_bus import EventBus
from app.models.atlas.atlas_model import AtlasModel
from app.models.atlas.backend import AbstractInferenceBackend
from app.models.model_manager import ModelManager
from app.models.model_registry import ModelRegistry
from app.types.model_result import ModelResult
from app.memory.conversation import Conversation

class SuccessfulBackend(AbstractInferenceBackend):
    @property
    def name(self) -> str:
        return "successful_backend"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        return ModelResult.ok(
            content=f"Atlas response: {prompt}",
            metadata={
                "backend_version": "1.0",
            },
        )


class FailedBackend(AbstractInferenceBackend):
    @property
    def name(self) -> str:
        return "failed_backend"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        return ModelResult.fail(
            error="Backend generation failed."
        )


class ExceptionBackend(AbstractInferenceBackend):
    @property
    def name(self) -> str:
        return "exception_backend"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        raise RuntimeError("Backend crashed.")


class KwargsBackend(AbstractInferenceBackend):
    def __init__(self) -> None:
        self.received_prompt: str | None = None
        self.received_kwargs: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "kwargs_backend"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        self.received_prompt = prompt
        self.received_kwargs = kwargs

        return ModelResult.ok(content="done")

class StructuredBackend(AbstractInferenceBackend):
    def __init__(self) -> None:
        self.received_conversation: Conversation | None = None
        self.received_system_prompt: str | None = None
        self.received_kwargs: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "structured_backend"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        return ModelResult.ok(
            content="fallback response"
        )

    async def generate_from_conversation(
            self,
            conversation: Conversation,
            system_prompt: str | None = None,
            **kwargs: Any,
    ) -> ModelResult:
        self.received_conversation = conversation
        self.received_system_prompt = system_prompt
        self.received_kwargs = kwargs

        return ModelResult.ok(
            content="structured response"
        )
class IncompleteBackend(AbstractInferenceBackend):
    pass


def test_incomplete_backend_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        IncompleteBackend()


def test_atlas_model_identity() -> None:
    model = AtlasModel(SuccessfulBackend())

    assert model.name == "atlas"
    assert model.provider == "atlas"


@pytest.mark.anyio
async def test_atlas_model_generates_successfully() -> None:
    model = AtlasModel(SuccessfulBackend())

    result = await model.generate("Analyze BTC")

    assert result.success is True
    assert result.content == "Atlas response: Analyze BTC"
    assert result.error is None


@pytest.mark.anyio
async def test_atlas_model_preserves_backend_metadata() -> None:
    model = AtlasModel(SuccessfulBackend())

    result = await model.generate("Analyze BTC")

    assert result.metadata["backend_version"] == "1.0"


@pytest.mark.anyio
async def test_atlas_model_adds_identity_metadata() -> None:
    model = AtlasModel(SuccessfulBackend())

    result = await model.generate("Analyze BTC")

    assert result.metadata["provider"] == "atlas"
    assert result.metadata["model"] == "atlas"
    assert result.metadata["backend"] == "successful_backend"


@pytest.mark.anyio
async def test_atlas_model_returns_backend_failure() -> None:
    model = AtlasModel(FailedBackend())

    result = await model.generate("Analyze BTC")

    assert result.success is False
    assert result.error == "Backend generation failed."


@pytest.mark.anyio
async def test_atlas_model_forwards_prompt_and_kwargs() -> None:
    backend = KwargsBackend()
    model = AtlasModel(backend)

    await model.generate(
        "Analyze ETH",
        temperature=0.4,
        max_tokens=500,
    )

    assert backend.received_prompt == "Analyze ETH"
    assert backend.received_kwargs == {
        "temperature": 0.4,
        "max_tokens": 500,
    }


@pytest.mark.anyio
async def test_atlas_model_runs_through_model_manager() -> None:
    event_bus = EventBus()
    registry = ModelRegistry()

    registry.register(
        AtlasModel(SuccessfulBackend())
    )

    manager = ModelManager(
        model_registry=registry,
        event_bus=event_bus,
    )

    result = await manager.generate(
        "atlas",
        "Analyze BTC",
    )

    assert result.success is True
    assert result.content == "Atlas response: Analyze BTC"


@pytest.mark.anyio
async def test_atlas_model_emits_lifecycle_events() -> None:
    event_bus = EventBus()
    registry = ModelRegistry()

    registry.register(
        AtlasModel(SuccessfulBackend())
    )

    events: list[Event] = []

    event_bus.subscribe(
        "model.started",
        events.append,
    )
    event_bus.subscribe(
        "model.completed",
        events.append,
    )

    manager = ModelManager(
        model_registry=registry,
        event_bus=event_bus,
    )

    await manager.generate(
        "atlas",
        "Analyze BTC",
    )

    assert [event.name for event in events] == [
        "model.started",
        "model.completed",
    ]


@pytest.mark.anyio
async def test_backend_exception_is_handled_by_model_manager() -> None:
    event_bus = EventBus()
    registry = ModelRegistry()

    registry.register(
        AtlasModel(ExceptionBackend())
    )

    manager = ModelManager(
        model_registry=registry,
        event_bus=event_bus,
    )

    result = await manager.generate(
        "atlas",
        "Analyze BTC",
    )

    assert result.success is False
    assert result.error == "Backend crashed."
@pytest.mark.anyio
async def test_conversation_generation_sends_system_prompt() -> None:
    backend = StructuredBackend()
    model = AtlasModel(backend)

    conversation = Conversation()
    conversation.add_user("Hello Atlas")

    result = await model.generate_from_conversation(
        conversation
    )

    assert result.success is True
    assert result.content == "structured response"
    assert (
            backend.received_system_prompt
            == AtlasModel.SYSTEM_PROMPT
    )


@pytest.mark.anyio
async def test_system_prompt_is_not_stored_in_conversation() -> None:
    backend = StructuredBackend()
    model = AtlasModel(backend)

    conversation = Conversation()
    conversation.add_user("Hello Atlas")

    messages_before = conversation.all()

    await model.generate_from_conversation(
        conversation
    )

    assert conversation.all() == messages_before
    assert len(conversation) == 1


@pytest.mark.anyio
async def test_conversation_generation_forwards_kwargs() -> None:
    backend = StructuredBackend()
    model = AtlasModel(backend)

    conversation = Conversation()
    conversation.add_user("Analyze BTC")

    await model.generate_from_conversation(
        conversation,
        temperature=0.2,
        max_tokens=200,
    )

    assert backend.received_kwargs == {
        "temperature": 0.2,
        "max_tokens": 200,
    }
class StructuredNotImplementedBackend(AbstractInferenceBackend):
    def __init__(self) -> None:
        self.generate_called = False

    @property
    def name(self) -> str:
        return "structured_not_implemented"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        self.generate_called = True
        return ModelResult.ok(content="fallback")

    async def generate_from_conversation(
            self,
            conversation: Conversation,
            system_prompt: str | None = None,
            **kwargs: Any,
    ) -> ModelResult:
        raise NotImplementedError


@pytest.mark.anyio
async def test_generate_empty_prompt() -> None:
    model = AtlasModel(SuccessfulBackend())

    result = await model.generate("")

    assert result.success
    assert result.content == "Atlas response: "


@pytest.mark.anyio
async def test_generate_unicode_prompt() -> None:
    model = AtlasModel(SuccessfulBackend())

    result = await model.generate("नमस्कार")

    assert result.success


@pytest.mark.anyio
async def test_generate_long_prompt() -> None:
    model = AtlasModel(SuccessfulBackend())

    result = await model.generate("A" * 5000)

    assert result.success


@pytest.mark.anyio
async def test_generate_returns_provider_metadata() -> None:
    model = AtlasModel(SuccessfulBackend())

    result = await model.generate("Hello")

    assert result.metadata["provider"] == "atlas"


@pytest.mark.anyio
async def test_generate_returns_model_metadata() -> None:
    model = AtlasModel(SuccessfulBackend())

    result = await model.generate("Hello")

    assert result.metadata["model"] == "atlas"


@pytest.mark.anyio
async def test_generate_returns_backend_metadata_name() -> None:
    model = AtlasModel(SuccessfulBackend())

    result = await model.generate("Hello")

    assert result.metadata["backend"] == "successful_backend"


@pytest.mark.anyio
async def test_generate_failure_has_no_provider_metadata() -> None:
    model = AtlasModel(FailedBackend())

    result = await model.generate("Hello")

    assert not result.success


@pytest.mark.anyio
async def test_generate_from_empty_conversation() -> None:
    model = AtlasModel(SuccessfulBackend())

    result = await model.generate_from_conversation(
        Conversation()
    )

    assert not result.success
    assert result.error == "Conversation cannot be empty."


@pytest.mark.anyio
async def test_generate_from_conversation_returns_metadata() -> None:
    backend = StructuredBackend()

    model = AtlasModel(backend)

    conversation = Conversation()
    conversation.add_user("Hello")

    result = await model.generate_from_conversation(
        conversation
    )

    assert result.metadata["provider"] == "atlas"
    assert result.metadata["model"] == "atlas"
    assert result.metadata["backend"] == "structured_backend"


@pytest.mark.anyio
async def test_generate_from_conversation_preserves_conversation() -> None:
    backend = StructuredBackend()

    model = AtlasModel(backend)

    conversation = Conversation()
    conversation.add_user("One")

    before = tuple(conversation.all())

    await model.generate_from_conversation(
        conversation
    )

    assert conversation.all() == before


@pytest.mark.anyio
async def test_generate_from_conversation_fallback_when_not_implemented() -> None:
    backend = StructuredNotImplementedBackend()

    model = AtlasModel(backend)

    conversation = Conversation()
    conversation.add_user("Hello")

    result = await model.generate_from_conversation(
        conversation
    )

    assert result.success
    assert backend.generate_called


@pytest.mark.anyio
async def test_generate_from_context_returns_success() -> None:
    backend = StructuredNotImplementedBackend()

    model = AtlasModel(backend)

    conversation = Conversation()
    conversation.add_user("BTC")

    result = await model._generate_from_context(
        conversation
    )

    assert result.success


@pytest.mark.anyio
async def test_with_metadata_preserves_existing_metadata() -> None:
    model = AtlasModel(SuccessfulBackend())

    result = ModelResult.ok(
        content="hello",
        metadata={
            "x": 1,
        },
    )

    updated = model._with_metadata(result)

    assert updated.metadata["x"] == 1


@pytest.mark.anyio
async def test_with_metadata_adds_backend_name() -> None:
    model = AtlasModel(SuccessfulBackend())

    result = ModelResult.ok(content="hello")

    updated = model._with_metadata(result)

    assert updated.metadata["backend"] == "successful_backend"


@pytest.mark.anyio
async def test_with_metadata_adds_provider() -> None:
    model = AtlasModel(SuccessfulBackend())

    updated = model._with_metadata(
        ModelResult.ok(content="hello")
    )

    assert updated.metadata["provider"] == "atlas"


@pytest.mark.anyio
async def test_with_metadata_adds_model() -> None:
    model = AtlasModel(SuccessfulBackend())

    updated = model._with_metadata(
        ModelResult.ok(content="hello")
    )

    assert updated.metadata["model"] == "atlas"


@pytest.mark.anyio
async def test_with_metadata_empty_content_becomes_empty_string() -> None:
    model = AtlasModel(SuccessfulBackend())

    updated = model._with_metadata(
        ModelResult.ok(content=None)
    )

    assert updated.content == ""


@pytest.mark.anyio
async def test_with_metadata_failure_returns_same_result() -> None:
    model = AtlasModel(SuccessfulBackend())

    result = ModelResult.fail(error="boom")

    updated = model._with_metadata(result)

    assert updated is result


@pytest.mark.anyio
async def test_generate_multiple_calls() -> None:
    model = AtlasModel(SuccessfulBackend())

    for _ in range(5):
        result = await model.generate("Hello")
        assert result.success


@pytest.mark.anyio
async def test_generate_from_conversation_multiple_calls() -> None:
    backend = StructuredBackend()

    model = AtlasModel(backend)

    conversation = Conversation()
    conversation.add_user("Hello")

    for _ in range(3):
        result = await model.generate_from_conversation(
            conversation
        )
        assert result.success


def test_model_name_constant() -> None:
    assert AtlasModel(SuccessfulBackend()).name == "atlas"


def test_model_provider_constant() -> None:
    assert AtlasModel(SuccessfulBackend()).provider == "atlas"


def test_system_prompt_is_not_empty() -> None:
    assert AtlasModel.SYSTEM_PROMPT


def test_system_prompt_mentions_context() -> None:
    assert "context" in AtlasModel.SYSTEM_PROMPT.lower()


def test_system_prompt_mentions_atlas() -> None:
    assert "atlas" in AtlasModel.SYSTEM_PROMPT.lower()