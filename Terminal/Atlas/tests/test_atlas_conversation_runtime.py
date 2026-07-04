from typing import Any

import pytest

from app.memory.conversation import Conversation
from app.models.atlas.atlas_model import AtlasModel
from app.models.atlas.backend import AbstractInferenceBackend
from app.types.model_result import ModelResult


class RecordingBackend(AbstractInferenceBackend):
    def __init__(self) -> None:
        self.received_prompt: str | None = None
        self.received_kwargs: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "recording_backend"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        self.received_prompt = prompt
        self.received_kwargs = kwargs

        return ModelResult.ok(
            content="Atlas generated a response."
        )


class FailedConversationBackend(AbstractInferenceBackend):
    @property
    def name(self) -> str:
        return "failed_conversation_backend"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        return ModelResult.fail(
            error="Conversation generation failed."
        )


@pytest.mark.anyio
async def test_generates_from_user_message() -> None:
    backend = RecordingBackend()
    model = AtlasModel(backend)

    conversation = Conversation()
    conversation.add_user("Hello Atlas")

    result = await model.generate_from_conversation(
        conversation
    )

    assert result.success is True
    assert result.content == "Atlas generated a response."


@pytest.mark.anyio
async def test_empty_conversation_returns_failure() -> None:
    model = AtlasModel(RecordingBackend())

    result = await model.generate_from_conversation(
        Conversation()
    )

    assert result.success is False
    assert result.error == "Conversation cannot be empty."


@pytest.mark.anyio
async def test_empty_conversation_does_not_call_backend() -> None:
    backend = RecordingBackend()
    model = AtlasModel(backend)

    await model.generate_from_conversation(
        Conversation()
    )

    assert backend.received_prompt is None


@pytest.mark.anyio
async def test_user_message_is_converted_to_context() -> None:
    backend = RecordingBackend()
    model = AtlasModel(backend)

    conversation = Conversation()
    conversation.add_user("Analyze BTC")

    await model.generate_from_conversation(
        conversation
    )

    assert backend.received_prompt == "User: Analyze BTC"


@pytest.mark.anyio
async def test_system_and_user_messages_are_forwarded() -> None:
    backend = RecordingBackend()
    model = AtlasModel(backend)

    conversation = Conversation()
    conversation.add_system("You are Atlas.")
    conversation.add_user("Analyze BTC")

    await model.generate_from_conversation(
        conversation
    )

    assert backend.received_prompt == (
        "System: You are Atlas.\n"
        "User: Analyze BTC"
    )


@pytest.mark.anyio
async def test_multi_turn_history_is_forwarded_in_order() -> None:
    backend = RecordingBackend()
    model = AtlasModel(backend)

    conversation = Conversation()
    conversation.add_user("Analyze BTC")
    conversation.add_assistant("BTC is bullish.")
    conversation.add_user("What about ETH?")

    await model.generate_from_conversation(
        conversation
    )

    assert backend.received_prompt == (
        "User: Analyze BTC\n"
        "Assistant: BTC is bullish.\n"
        "User: What about ETH?"
    )


@pytest.mark.anyio
async def test_tool_messages_are_included() -> None:
    backend = RecordingBackend()
    model = AtlasModel(backend)

    conversation = Conversation()
    conversation.add_user("Get BTC price")
    conversation.add_tool("BTC price is 65000")

    await model.generate_from_conversation(
        conversation
    )

    assert backend.received_prompt == (
        "User: Get BTC price\n"
        "Tool: BTC price is 65000"
    )


@pytest.mark.anyio
async def test_generation_kwargs_are_forwarded() -> None:
    backend = RecordingBackend()
    model = AtlasModel(backend)

    conversation = Conversation()
    conversation.add_user("Analyze BTC")

    await model.generate_from_conversation(
        conversation,
        temperature=0.3,
        max_tokens=500,
    )

    assert backend.received_kwargs == {
        "temperature": 0.3,
        "max_tokens": 500,
    }


@pytest.mark.anyio
async def test_backend_failure_is_preserved() -> None:
    model = AtlasModel(
        FailedConversationBackend()
    )

    conversation = Conversation()
    conversation.add_user("Analyze BTC")

    result = await model.generate_from_conversation(
        conversation
    )

    assert result.success is False
    assert result.error == "Conversation generation failed."


@pytest.mark.anyio
async def test_successful_result_contains_atlas_metadata() -> None:
    model = AtlasModel(RecordingBackend())

    conversation = Conversation()
    conversation.add_user("Hello")

    result = await model.generate_from_conversation(
        conversation
    )

    assert result.metadata["provider"] == "atlas"
    assert result.metadata["model"] == "atlas"
    assert result.metadata["backend"] == "recording_backend"


@pytest.mark.anyio
async def test_generation_does_not_modify_conversation() -> None:
    backend = RecordingBackend()
    model = AtlasModel(backend)

    conversation = Conversation()
    message = conversation.add_user("Hello Atlas")

    await model.generate_from_conversation(
        conversation
    )

    assert conversation.all() == (message,)
    assert len(conversation) == 1


@pytest.mark.anyio
async def test_repeated_generation_uses_current_history() -> None:
    backend = RecordingBackend()
    model = AtlasModel(backend)

    conversation = Conversation()
    conversation.add_user("First question")

    await model.generate_from_conversation(
        conversation
    )

    assert backend.received_prompt == (
        "User: First question"
    )

    conversation.add_assistant("First answer")
    conversation.add_user("Second question")

    await model.generate_from_conversation(
        conversation
    )

    assert backend.received_prompt == (
        "User: First question\n"
        "Assistant: First answer\n"
        "User: Second question"
    )