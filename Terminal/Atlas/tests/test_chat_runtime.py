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
@pytest.mark.anyio
async def test_send_returns_same_result_instance() -> None:
    conversation = Conversation()
    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    result = await runtime.send(
        conversation,
        "Hello",
    )

    assert result is not None
    assert isinstance(result, ModelResult)


@pytest.mark.anyio
async def test_send_preserves_existing_messages() -> None:
    conversation = Conversation()
    conversation.add_user("First")
    conversation.add_assistant("Reply")

    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    await runtime.send(
        conversation,
        "Second",
    )

    assert conversation.all()[0].content == "First"
    assert conversation.all()[1].content == "Reply"


@pytest.mark.anyio
async def test_send_with_empty_kwargs() -> None:
    backend = RecordingChatBackend()
    runtime = ChatRuntime(
        AtlasModel(backend)
    )

    await runtime.send(
        Conversation(),
        "Hello",
    )

    assert backend.received_kwargs[0] == {}


@pytest.mark.anyio
async def test_send_accepts_unicode_message() -> None:
    conversation = Conversation()
    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    await runtime.send(
        conversation,
        "नमस्कार Atlas 🚀",
    )

    assert conversation.all()[0].content == "नमस्कार Atlas 🚀"


@pytest.mark.anyio
async def test_send_accepts_multiline_message() -> None:
    conversation = Conversation()

    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    message = "Line1\nLine2\nLine3"

    await runtime.send(
        conversation,
        message,
    )

    assert conversation.all()[0].content == message


@pytest.mark.anyio
async def test_send_accepts_long_message() -> None:
    conversation = Conversation()

    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    message = "A" * 5000

    await runtime.send(
        conversation,
        message,
    )

    assert conversation.all()[0].content == message


@pytest.mark.anyio
async def test_send_keeps_user_message_order() -> None:
    conversation = Conversation()

    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    await runtime.send(conversation, "One")
    await runtime.send(conversation, "Two")

    users = [
        m.content
        for m in conversation.all()
        if m.role is MessageRole.USER
    ]

    assert users == [
        "One",
        "Two",
    ]


@pytest.mark.anyio
async def test_send_keeps_assistant_message_order() -> None:
    conversation = Conversation()

    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    await runtime.send(conversation, "One")
    await runtime.send(conversation, "Two")

    assistants = [
        m.content
        for m in conversation.all()
        if m.role is MessageRole.ASSISTANT
    ]

    assert assistants == [
        "Atlas response.",
        "Atlas response.",
    ]


@pytest.mark.anyio
async def test_conversation_length_after_three_messages() -> None:
    conversation = Conversation()

    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    await runtime.send(conversation, "1")
    await runtime.send(conversation, "2")
    await runtime.send(conversation, "3")

    assert len(conversation) == 6


@pytest.mark.anyio
async def test_send_does_not_remove_previous_assistant_messages() -> None:
    conversation = Conversation()

    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    await runtime.send(conversation, "Hello")
    await runtime.send(conversation, "Again")

    assert conversation.all()[1].content == "Atlas response."


@pytest.mark.anyio
async def test_backend_receives_only_one_call_per_send() -> None:
    backend = RecordingChatBackend()

    runtime = ChatRuntime(
        AtlasModel(backend)
    )

    await runtime.send(
        Conversation(),
        "Hello",
    )

    assert len(backend.received_prompts) == 1


@pytest.mark.anyio
async def test_backend_receives_two_calls_after_two_sends() -> None:
    backend = RecordingChatBackend()

    runtime = ChatRuntime(
        AtlasModel(backend)
    )

    conversation = Conversation()

    await runtime.send(conversation, "One")
    await runtime.send(conversation, "Two")

    assert len(backend.received_prompts) == 2


@pytest.mark.anyio
async def test_last_message_after_success_is_assistant() -> None:
    conversation = Conversation()

    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    await runtime.send(
        conversation,
        "Hello",
    )

    assert conversation.last().role is MessageRole.ASSISTANT


@pytest.mark.anyio
async def test_last_message_after_failure_is_user() -> None:
    conversation = Conversation()

    runtime = ChatRuntime(
        AtlasModel(FailedChatBackend())
    )

    await runtime.send(
        conversation,
        "Hello",
    )

    assert conversation.last().role is MessageRole.USER


@pytest.mark.anyio
async def test_send_returns_success_every_time() -> None:
    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    conversation = Conversation()

    for _ in range(5):
        result = await runtime.send(
            conversation,
            "Hello",
        )

        assert result.success


@pytest.mark.anyio
async def test_conversation_is_same_instance_after_send() -> None:
    conversation = Conversation()

    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    original = id(conversation)

    await runtime.send(
        conversation,
        "Hello",
    )

    assert id(conversation) == original


@pytest.mark.anyio
async def test_backend_prompt_contains_latest_message() -> None:
    backend = RecordingChatBackend()

    runtime = ChatRuntime(
        AtlasModel(backend)
    )

    await runtime.send(
        Conversation(),
        "Latest",
    )

    assert "Latest" in backend.received_prompts[0]


@pytest.mark.anyio
async def test_failed_backend_returns_unsuccessful_result() -> None:
    runtime = ChatRuntime(
        AtlasModel(FailedChatBackend())
    )

    result = await runtime.send(
        Conversation(),
        "Hello",
    )

    assert not result.success


@pytest.mark.anyio
async def test_empty_response_backend_returns_success() -> None:
    runtime = ChatRuntime(
        AtlasModel(EmptyResponseBackend())
    )

    result = await runtime.send(
        Conversation(),
        "Hello",
    )

    assert result.success


@pytest.mark.anyio
async def test_empty_response_backend_keeps_single_message() -> None:
    conversation = Conversation()

    runtime = ChatRuntime(
        AtlasModel(EmptyResponseBackend())
    )

    await runtime.send(
        conversation,
        "Hello",
    )

    assert len(conversation.all()) == 1
from uuid import UUID

from app.memory.conversation_manager import ConversationManager


@pytest.mark.anyio
async def test_create_conversation_returns_uuid_and_conversation() -> None:
    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    conversation_id, conversation = runtime.create_conversation()

    assert isinstance(conversation_id, UUID)
    assert isinstance(conversation, Conversation)


@pytest.mark.anyio
async def test_create_conversation_is_empty() -> None:
    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    _, conversation = runtime.create_conversation()

    assert len(conversation) == 0


@pytest.mark.anyio
async def test_create_conversation_returns_unique_ids() -> None:
    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    id1, _ = runtime.create_conversation()
    id2, _ = runtime.create_conversation()

    assert id1 != id2


@pytest.mark.anyio
async def test_send_to_existing_conversation() -> None:
    manager = ConversationManager()

    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend()),
        manager,
    )

    conversation_id, conversation = manager.create()

    await runtime.send_to(
        conversation_id,
        "Hello",
    )

    assert len(conversation) == 2
    assert conversation.last().role is MessageRole.ASSISTANT


@pytest.mark.anyio
async def test_send_to_updates_correct_conversation() -> None:
    manager = ConversationManager()

    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend()),
        manager,
    )

    id1, conversation1 = manager.create()
    _, conversation2 = manager.create()

    await runtime.send_to(
        id1,
        "Hello",
    )

    assert len(conversation1) == 2
    assert len(conversation2) == 0


@pytest.mark.anyio
async def test_send_to_multiple_times_grows_history() -> None:
    manager = ConversationManager()

    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend()),
        manager,
    )

    conversation_id, conversation = manager.create()

    await runtime.send_to(conversation_id, "One")
    await runtime.send_to(conversation_id, "Two")

    assert len(conversation) == 4


@pytest.mark.anyio
async def test_send_to_invalid_id_raises() -> None:
    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend())
    )

    with pytest.raises(KeyError):
        await runtime.send_to(
            UUID(int=999),
            "Hello",
        )


def test_create_title_simple() -> None:
    assert (
            ChatRuntime._create_title("Hello Atlas")
            == "Hello Atlas"
    )


def test_create_title_strips_spaces() -> None:
    assert (
            ChatRuntime._create_title(
                "    Hello Atlas     "
            )
            == "Hello Atlas"
    )


def test_create_title_collapses_multiple_spaces() -> None:
    assert (
            ChatRuntime._create_title(
                "Hello      Atlas"
            )
            == "Hello Atlas"
    )


def test_create_title_collapses_tabs() -> None:
    assert (
            ChatRuntime._create_title(
                "Hello\t\tAtlas"
            )
            == "Hello Atlas"
    )


def test_create_title_collapses_newlines() -> None:
    assert (
            ChatRuntime._create_title(
                "Hello\n\nAtlas"
            )
            == "Hello Atlas"
    )


def test_create_title_exact_limit() -> None:
    text = "A" * 50

    assert ChatRuntime._create_title(text) == text


def test_create_title_truncates_long_text() -> None:
    text = "A" * 80

    title = ChatRuntime._create_title(text)

    assert title.endswith("...")
    assert len(title) == 53


def test_create_title_custom_limit() -> None:
    title = ChatRuntime._create_title(
        "abcdefghijklmnopqrstuvwxyz",
        max_length=10,
    )

    assert title == "abcdefghij..."


def test_create_title_empty_string() -> None:
    assert ChatRuntime._create_title("") == ""


def test_create_title_only_spaces() -> None:
    assert ChatRuntime._create_title("      ") == ""


def test_create_title_unicode() -> None:
    assert (
            ChatRuntime._create_title("नमस्कार Atlas")
            == "नमस्कार Atlas"
    )


def test_create_title_emojis() -> None:
    assert (
            ChatRuntime._create_title("🚀 Atlas")
            == "🚀 Atlas"
    )


@pytest.mark.anyio
async def test_send_to_returns_model_result() -> None:
    manager = ConversationManager()

    runtime = ChatRuntime(
        AtlasModel(RecordingChatBackend()),
        manager,
    )

    conversation_id, _ = manager.create()

    result = await runtime.send_to(
        conversation_id,
        "Hello",
    )

    assert result.success