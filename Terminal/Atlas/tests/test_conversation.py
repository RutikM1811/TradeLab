from dataclasses import FrozenInstanceError
from datetime import UTC
from uuid import UUID

import pytest

from app.memory.conversation import Conversation
from app.types.message import Message, MessageRole


def test_message_roles_have_expected_values() -> None:
    assert MessageRole.SYSTEM == "system"
    assert MessageRole.USER == "user"
    assert MessageRole.ASSISTANT == "assistant"
    assert MessageRole.TOOL == "tool"


def test_message_has_unique_id() -> None:
    first = Message(
        role=MessageRole.USER,
        content="Hello",
    )
    second = Message(
        role=MessageRole.USER,
        content="Hello",
    )

    assert isinstance(first.id, UUID)
    assert first.id != second.id


def test_message_timestamp_is_utc() -> None:
    message = Message(
        role=MessageRole.USER,
        content="Hello",
    )

    assert message.created_at.tzinfo is UTC


def test_message_is_immutable() -> None:
    message = Message(
        role=MessageRole.USER,
        content="Hello",
    )

    with pytest.raises(FrozenInstanceError):
        message.content = "Changed"


def test_empty_conversation_has_no_last_message() -> None:
    conversation = Conversation()

    assert len(conversation) == 0
    assert conversation.last() is None
    assert conversation.all() == ()


def test_conversation_adds_message() -> None:
    conversation = Conversation()

    message = Message(
        role=MessageRole.USER,
        content="Hello Atlas",
    )

    conversation.add(message)

    assert len(conversation) == 1
    assert conversation.last() is message


def test_conversation_rejects_empty_content() -> None:
    conversation = Conversation()

    message = Message(
        role=MessageRole.USER,
        content="   ",
    )

    with pytest.raises(
            ValueError,
            match="Message content cannot be empty",
    ):
        conversation.add(message)


def test_add_system_creates_system_message() -> None:
    conversation = Conversation()

    message = conversation.add_system(
        "You are Atlas."
    )

    assert message.role is MessageRole.SYSTEM
    assert message.content == "You are Atlas."


def test_add_user_creates_user_message() -> None:
    conversation = Conversation()

    message = conversation.add_user(
        "Analyze BTC"
    )

    assert message.role is MessageRole.USER
    assert message.content == "Analyze BTC"


def test_add_assistant_creates_assistant_message() -> None:
    conversation = Conversation()

    message = conversation.add_assistant(
        "BTC is trending upward."
    )

    assert message.role is MessageRole.ASSISTANT


def test_add_tool_creates_tool_message() -> None:
    conversation = Conversation()

    message = conversation.add_tool(
        "BTCUSDT price: 65000"
    )

    assert message.role is MessageRole.TOOL


def test_conversation_preserves_message_order() -> None:
    conversation = Conversation()

    system = conversation.add_system(
        "You are Atlas."
    )
    user = conversation.add_user(
        "Analyze BTC"
    )
    assistant = conversation.add_assistant(
        "Analyzing BTC."
    )

    assert conversation.all() == (
        system,
        user,
        assistant,
    )


def test_last_returns_most_recent_message() -> None:
    conversation = Conversation()

    conversation.add_user("First")
    latest = conversation.add_assistant("Second")

    assert conversation.last() is latest


def test_all_returns_immutable_tuple() -> None:
    conversation = Conversation()

    conversation.add_user("Hello")

    messages = conversation.all()

    assert isinstance(messages, tuple)


def test_message_supports_metadata() -> None:
    message = Message(
        role=MessageRole.TOOL,
        content="Price fetched.",
        metadata={
            "tool_name": "get_price",
            "symbol": "BTCUSDT",
        },
    )

    assert message.metadata["tool_name"] == "get_price"
    assert message.metadata["symbol"] == "BTCUSDT"