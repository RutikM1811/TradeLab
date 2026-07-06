from typing import Any
from uuid import uuid4

import pytest

from app.memory.conversation_manager import ConversationManager
from app.models.atlas.atlas_model import AtlasModel
from app.models.atlas.backend import AbstractInferenceBackend
from app.services.chat_runtime import ChatRuntime
from app.types.message import MessageRole
from app.types.model_result import ModelResult


class ManagedChatBackend(AbstractInferenceBackend):
    def __init__(self) -> None:
        self.received_prompts: list[str] = []
        self.received_kwargs: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "managed_chat_backend"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        self.received_prompts.append(prompt)
        self.received_kwargs.append(kwargs)

        return ModelResult.ok(
            content="Atlas response."
        )


class FailedManagedChatBackend(AbstractInferenceBackend):
    @property
    def name(self) -> str:
        return "failed_managed_chat_backend"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        return ModelResult.fail(
            error="Managed chat failed."
        )


def test_create_conversation_returns_managed_session() -> None:
    manager = ConversationManager()
    runtime = ChatRuntime(
        AtlasModel(ManagedChatBackend()),
        manager,
    )

    conversation_id, conversation = (
        runtime.create_conversation()
    )

    assert manager.contains(conversation_id)
    assert manager.get(conversation_id) is conversation


def test_multiple_created_sessions_have_unique_ids() -> None:
    runtime = ChatRuntime(
        AtlasModel(ManagedChatBackend())
    )

    first_id, _ = runtime.create_conversation()
    second_id, _ = runtime.create_conversation()

    assert first_id != second_id


@pytest.mark.anyio
async def test_send_to_adds_user_and_assistant_messages() -> None:
    runtime = ChatRuntime(
        AtlasModel(ManagedChatBackend())
    )

    conversation_id, conversation = (
        runtime.create_conversation()
    )

    await runtime.send_to(
        conversation_id,
        "Hello Atlas",
    )

    assert len(conversation) == 2
    assert conversation.all()[0].role is MessageRole.USER
    assert conversation.all()[1].role is MessageRole.ASSISTANT


@pytest.mark.anyio
async def test_send_to_returns_model_result() -> None:
    runtime = ChatRuntime(
        AtlasModel(ManagedChatBackend())
    )

    conversation_id, _ = runtime.create_conversation()

    result = await runtime.send_to(
        conversation_id,
        "Hello Atlas",
    )

    assert result.success is True
    assert result.content == "Atlas response."


@pytest.mark.anyio
async def test_send_to_passes_message_to_backend() -> None:
    backend = ManagedChatBackend()
    runtime = ChatRuntime(
        AtlasModel(backend)
    )

    conversation_id, _ = runtime.create_conversation()

    await runtime.send_to(
        conversation_id,
        "Analyze BTC",
    )

    assert backend.received_prompts[0] == (
        "User: Analyze BTC"
    )


@pytest.mark.anyio
async def test_send_to_preserves_multi_turn_context() -> None:
    backend = ManagedChatBackend()
    runtime = ChatRuntime(
        AtlasModel(backend)
    )

    conversation_id, _ = runtime.create_conversation()

    await runtime.send_to(
        conversation_id,
        "First question",
    )

    await runtime.send_to(
        conversation_id,
        "Second question",
    )

    assert backend.received_prompts[1] == (
        "User: First question\n"
        "Assistant: Atlas response.\n"
        "User: Second question"
    )


@pytest.mark.anyio
async def test_send_to_grows_managed_history() -> None:
    runtime = ChatRuntime(
        AtlasModel(ManagedChatBackend())
    )

    conversation_id, conversation = (
        runtime.create_conversation()
    )

    await runtime.send_to(
        conversation_id,
        "First",
    )

    await runtime.send_to(
        conversation_id,
        "Second",
    )

    assert len(conversation) == 4


@pytest.mark.anyio
async def test_separate_sessions_are_isolated() -> None:
    backend = ManagedChatBackend()
    runtime = ChatRuntime(
        AtlasModel(backend)
    )

    first_id, first = runtime.create_conversation()
    second_id, second = runtime.create_conversation()

    await runtime.send_to(
        first_id,
        "Message for first",
    )

    await runtime.send_to(
        second_id,
        "Message for second",
    )

    assert len(first) == 2
    assert len(second) == 2

    assert first.all()[0].content == "Message for first"
    assert second.all()[0].content == "Message for second"


@pytest.mark.anyio
async def test_second_session_does_not_receive_first_history() -> None:
    backend = ManagedChatBackend()
    runtime = ChatRuntime(
        AtlasModel(backend)
    )

    first_id, _ = runtime.create_conversation()
    second_id, _ = runtime.create_conversation()

    await runtime.send_to(
        first_id,
        "Private first message",
    )

    await runtime.send_to(
        second_id,
        "Second session message",
    )

    assert backend.received_prompts[1] == (
        "User: Second session message"
    )


@pytest.mark.anyio
async def test_send_to_forwards_generation_kwargs() -> None:
    backend = ManagedChatBackend()
    runtime = ChatRuntime(
        AtlasModel(backend)
    )

    conversation_id, _ = runtime.create_conversation()

    await runtime.send_to(
        conversation_id,
        "Analyze BTC",
        temperature=0.2,
        max_tokens=400,
    )

    assert backend.received_kwargs[0] == {
        "temperature": 0.2,
        "max_tokens": 400,
    }


@pytest.mark.anyio
async def test_send_to_raises_for_missing_conversation() -> None:
    runtime = ChatRuntime(
        AtlasModel(ManagedChatBackend())
    )

    with pytest.raises(
            KeyError,
            match="was not found",
    ):
        await runtime.send_to(
            uuid4(),
            "Hello",
        )


@pytest.mark.anyio
async def test_missing_conversation_does_not_call_backend() -> None:
    backend = ManagedChatBackend()
    runtime = ChatRuntime(
        AtlasModel(backend)
    )

    with pytest.raises(KeyError):
        await runtime.send_to(
            uuid4(),
            "Hello",
        )

    assert backend.received_prompts == []


@pytest.mark.anyio
async def test_failed_send_preserves_user_message() -> None:
    runtime = ChatRuntime(
        AtlasModel(FailedManagedChatBackend())
    )

    conversation_id, conversation = (
        runtime.create_conversation()
    )

    result = await runtime.send_to(
        conversation_id,
        "Important question",
    )

    assert result.success is False
    assert len(conversation) == 1
    assert conversation.last() is not None
    assert conversation.last().role is MessageRole.USER
    assert conversation.last().content == "Important question"


@pytest.mark.anyio
async def test_failed_send_does_not_add_assistant_message() -> None:
    runtime = ChatRuntime(
        AtlasModel(FailedManagedChatBackend())
    )

    conversation_id, conversation = (
        runtime.create_conversation()
    )

    await runtime.send_to(
        conversation_id,
        "Hello",
    )

    assert len(conversation) == 1


@pytest.mark.anyio
async def test_runtime_uses_injected_conversation_manager() -> None:
    manager = ConversationManager()

    existing_id, existing_conversation = manager.create()
    existing_conversation.add_system("You are Atlas.")

    runtime = ChatRuntime(
        AtlasModel(ManagedChatBackend()),
        manager,
    )

    await runtime.send_to(
        existing_id,
        "Hello",
    )

    assert len(existing_conversation) == 3
    assert existing_conversation.all()[0].role is MessageRole.SYSTEM