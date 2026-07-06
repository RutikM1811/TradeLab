from typing import Any

import pytest

from app.memory.conversation import Conversation
from app.models.atlas.atlas_model import AtlasModel
from app.models.atlas.backend import AbstractInferenceBackend
from app.services.chat_runtime import ChatRuntime
from app.types.message import MessageRole
from app.types.model_result import ModelResult


class RecordingChatBackend(AbstractInferenceBackend):
    def __init__(self) -> None:
        self.received_prompts: list[str] = []
        self.received_kwargs: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "recording_chat_backend"

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


class FailedChatBackend(AbstractInferenceBackend):
    @property
    def name(self) -> str:
        return "failed_chat_backend"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        return ModelResult.fail(
            error="Chat generation failed."
        )


class EmptyResponseBackend(AbstractInferenceBackend):
    @property
    def name(self) -> str:
        return "empty_response_backend"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        return ModelResult.ok(content="")


@pytest.mark.anyio
async def test_send_adds_user_message() -> None:
    conversation = Conversation()
    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    await runtime.send(
        conversation,
        "Hello Atlas",
    )

    assert conversation.all()[0].role is MessageRole.USER
    assert conversation.all()[0].content == "Hello Atlas"


@pytest.mark.anyio
async def test_send_adds_assistant_response() -> None:
    conversation = Conversation()
    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    await runtime.send(
        conversation,
        "Hello Atlas",
    )

    assert conversation.all()[1].role is MessageRole.ASSISTANT
    assert conversation.all()[1].content == "Atlas response."


@pytest.mark.anyio
async def test_successful_send_returns_model_result() -> None:
    conversation = Conversation()
    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    result = await runtime.send(
        conversation,
        "Hello Atlas",
    )

    assert result.success is True
    assert result.content == "Atlas response."


@pytest.mark.anyio
async def test_single_send_adds_two_messages() -> None:
    conversation = Conversation()
    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    await runtime.send(
        conversation,
        "Hello Atlas",
    )

    assert len(conversation) == 2


@pytest.mark.anyio
async def test_existing_system_message_is_preserved() -> None:
    conversation = Conversation()
    conversation.add_system("You are Atlas.")

    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    await runtime.send(
        conversation,
        "Hello",
    )

    assert conversation.all()[0].role is MessageRole.SYSTEM
    assert conversation.all()[0].content == "You are Atlas."
    assert len(conversation) == 3


@pytest.mark.anyio
async def test_backend_receives_current_user_message() -> None:
    backend = RecordingChatBackend()
    runtime = ChatRuntime(
        AtlasModel(backend)
    )
    conversation = Conversation()

    await runtime.send(
        conversation,
        "Analyze BTC",
    )

    assert backend.received_prompts[0] == (
        "User: Analyze BTC"
    )


@pytest.mark.anyio
async def test_backend_receives_full_existing_history() -> None:
    backend = RecordingChatBackend()
    runtime = ChatRuntime(
        AtlasModel(backend)
    )

    conversation = Conversation()
    conversation.add_system("You are Atlas.")
    conversation.add_user("First question")
    conversation.add_assistant("First answer")

    await runtime.send(
        conversation,
        "Second question",
    )

    assert backend.received_prompts[0] == (
        "System: You are Atlas.\n"
        "User: First question\n"
        "Assistant: First answer\n"
        "User: Second question"
    )


@pytest.mark.anyio
async def test_multiple_sends_preserve_context_continuity() -> None:
    backend = RecordingChatBackend()
    runtime = ChatRuntime(
        AtlasModel(backend)
    )
    conversation = Conversation()

    await runtime.send(
        conversation,
        "First question",
    )

    await runtime.send(
        conversation,
        "Second question",
    )

    assert backend.received_prompts[1] == (
        "User: First question\n"
        "Assistant: Atlas response.\n"
        "User: Second question"
    )


@pytest.mark.anyio
async def test_multiple_sends_grow_conversation_history() -> None:
    conversation = Conversation()
    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    await runtime.send(
        conversation,
        "First question",
    )

    await runtime.send(
        conversation,
        "Second question",
    )

    assert len(conversation) == 4


@pytest.mark.anyio
async def test_generation_kwargs_are_forwarded() -> None:
    backend = RecordingChatBackend()
    runtime = ChatRuntime(
        AtlasModel(backend)
    )
    conversation = Conversation()

    await runtime.send(
        conversation,
        "Analyze BTC",
        temperature=0.2,
        max_tokens=300,
    )

    assert backend.received_kwargs[0] == {
        "temperature": 0.2,
        "max_tokens": 300,
    }


@pytest.mark.anyio
async def test_failed_generation_does_not_add_assistant_message() -> None:
    conversation = Conversation()
    runtime = ChatRuntime(
        AtlasModel(FailedChatBackend())
    )

    result = await runtime.send(
        conversation,
        "Hello Atlas",
    )

    assert result.success is False
    assert len(conversation) == 1
    assert conversation.last() is not None
    assert conversation.last().role is MessageRole.USER


@pytest.mark.anyio
async def test_failed_generation_preserves_user_message() -> None:
    conversation = Conversation()
    runtime = ChatRuntime(
        AtlasModel(FailedChatBackend())
    )

    await runtime.send(
        conversation,
        "Important question",
    )

    assert conversation.last() is not None
    assert conversation.last().content == "Important question"


@pytest.mark.anyio
async def test_failed_generation_returns_error() -> None:
    conversation = Conversation()
    runtime = ChatRuntime(
        AtlasModel(FailedChatBackend())
    )

    result = await runtime.send(
        conversation,
        "Hello",
    )

    assert result.error == "Chat generation failed."


@pytest.mark.anyio
async def test_empty_assistant_response_is_not_stored() -> None:
    conversation = Conversation()
    runtime = ChatRuntime(
        AtlasModel(EmptyResponseBackend())
    )

    result = await runtime.send(
        conversation,
        "Hello",
    )

    assert result.success is True
    assert result.content == ""

    # Only the user message should remain.
    assert len(conversation) == 1
    assert conversation.last() is not None
    assert conversation.last().role is MessageRole.USER
    assert conversation.last().content == "Hello"


@pytest.mark.anyio
async def test_tool_history_is_included_in_chat_context() -> None:
    backend = RecordingChatBackend()
    runtime = ChatRuntime(
        AtlasModel(backend)
    )

    conversation = Conversation()
    conversation.add_user("Get BTC price")
    conversation.add_tool("BTC price is 65000")

    await runtime.send(
        conversation,
        "What does that mean?",
    )

    assert backend.received_prompts[0] == (
        "User: Get BTC price\n"
        "Tool: BTC price is 65000\n"
        "User: What does that mean?"
    )