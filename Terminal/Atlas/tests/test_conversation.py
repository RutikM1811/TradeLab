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
def test_new_conversation_has_default_metadata() -> None:
    conversation = Conversation()

    assert conversation.metadata is not None


def test_add_returns_none() -> None:
    conversation = Conversation()

    message = Message(
        role=MessageRole.USER,
        content="Hello",
    )

    result = conversation.add(message)

    assert result is None


def test_add_preserves_same_message_instance() -> None:
    conversation = Conversation()

    message = Message(
        role=MessageRole.USER,
        content="Hello",
    )

    conversation.add(message)

    assert conversation.all()[0] is message


def test_len_increases_after_each_add() -> None:
    conversation = Conversation()

    assert len(conversation) == 0

    conversation.add_user("One")
    assert len(conversation) == 1

    conversation.add_user("Two")
    assert len(conversation) == 2

    conversation.add_user("Three")
    assert len(conversation) == 3


def test_add_user_returns_last_message() -> None:
    conversation = Conversation()

    message = conversation.add_user("Hello")

    assert conversation.last() is message


def test_add_system_returns_last_message() -> None:
    conversation = Conversation()

    message = conversation.add_system("System")

    assert conversation.last() is message


def test_add_assistant_returns_last_message() -> None:
    conversation = Conversation()

    message = conversation.add_assistant("Reply")

    assert conversation.last() is message


def test_add_tool_returns_last_message() -> None:
    conversation = Conversation()

    message = conversation.add_tool("Tool output")

    assert conversation.last() is message


def test_multiple_user_messages_are_preserved() -> None:
    conversation = Conversation()

    conversation.add_user("One")
    conversation.add_user("Two")
    conversation.add_user("Three")

    assert [m.content for m in conversation.all()] == [
        "One",
        "Two",
        "Three",
    ]


def test_multiple_roles_are_preserved() -> None:
    conversation = Conversation()

    conversation.add_system("S")
    conversation.add_user("U")
    conversation.add_tool("T")
    conversation.add_assistant("A")

    assert [m.role for m in conversation.all()] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]


def test_last_after_single_message() -> None:
    conversation = Conversation()

    message = conversation.add_user("Hello")

    assert conversation.last() is message


def test_all_on_empty_conversation_returns_empty_tuple() -> None:
    conversation = Conversation()

    assert conversation.all() == ()


def test_add_user_accepts_unicode() -> None:
    conversation = Conversation()

    message = conversation.add_user("नमस्कार")

    assert message.content == "नमस्कार"


def test_add_user_accepts_emoji() -> None:
    conversation = Conversation()

    message = conversation.add_user("🚀")

    assert message.content == "🚀"


def test_add_user_accepts_multiline() -> None:
    conversation = Conversation()

    text = "One\nTwo\nThree"

    message = conversation.add_user(text)

    assert message.content == text


def test_add_user_accepts_long_text() -> None:
    conversation = Conversation()

    text = "A" * 10000

    message = conversation.add_user(text)

    assert message.content == text


def test_message_role_is_preserved_after_add() -> None:
    conversation = Conversation()

    message = Message(
        role=MessageRole.TOOL,
        content="Output",
    )

    conversation.add(message)

    assert conversation.last().role is MessageRole.TOOL


def test_message_content_is_preserved_after_add() -> None:
    conversation = Conversation()

    message = Message(
        role=MessageRole.USER,
        content="Exact text",
    )

    conversation.add(message)

    assert conversation.last().content == "Exact text"


def test_message_metadata_is_preserved() -> None:
    conversation = Conversation()

    message = Message(
        role=MessageRole.USER,
        content="Hello",
        metadata={"key": "value"},
    )

    conversation.add(message)

    assert conversation.last().metadata["key"] == "value"


def test_last_changes_after_every_addition() -> None:
    conversation = Conversation()

    first = conversation.add_user("One")
    second = conversation.add_assistant("Two")
    third = conversation.add_tool("Three")

    assert conversation.last() is third
    assert conversation.last() is not second
    assert conversation.last() is not first


def test_conversation_keeps_exact_message_count() -> None:
    conversation = Conversation()

    for i in range(10):
        conversation.add_user(str(i))

    assert len(conversation) == 10


def test_message_ids_are_all_unique_in_conversation() -> None:
    conversation = Conversation()

    for i in range(5):
        conversation.add_user(str(i))

    ids = [m.id for m in conversation.all()]

    assert len(ids) == len(set(ids))


def test_last_on_empty_conversation_remains_none() -> None:
    conversation = Conversation()

    assert conversation.last() is None
    assert conversation.last() is None


def test_all_returns_new_tuple_each_time() -> None:
    conversation = Conversation()

    conversation.add_user("Hello")

    first = conversation.all()
    second = conversation.all()

    assert first == second
    assert first is not second


def test_adding_message_does_not_modify_previous_messages() -> None:
    conversation = Conversation()

    first = conversation.add_user("One")

    conversation.add_assistant("Two")

    assert first.content == "One"
    assert first.role is MessageRole.USER