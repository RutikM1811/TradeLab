from datetime import UTC

from app.memory.conversation import Conversation
from app.memory.conversation_metadata import ConversationMetadata


def test_conversation_has_default_metadata() -> None:
    conversation = Conversation()

    assert isinstance(
        conversation.metadata,
        ConversationMetadata,
    )


def test_conversation_has_default_title() -> None:
    conversation = Conversation()

    assert conversation.metadata.title == "New Conversation"


def test_conversation_accepts_custom_metadata() -> None:
    metadata = ConversationMetadata(
        title="BTC Analysis"
    )

    conversation = Conversation(
        metadata=metadata
    )

    assert conversation.metadata is metadata
    assert conversation.metadata.title == "BTC Analysis"


def test_conversations_have_separate_metadata() -> None:
    first = Conversation()
    second = Conversation()

    first.metadata.rename("First Chat")

    assert first.metadata.title == "First Chat"
    assert second.metadata.title == "New Conversation"


def test_conversation_metadata_timestamps_are_utc() -> None:
    conversation = Conversation()

    assert conversation.metadata.created_at.tzinfo is UTC
    assert conversation.metadata.updated_at.tzinfo is UTC


def test_adding_user_message_updates_metadata() -> None:
    conversation = Conversation()
    original_updated_at = conversation.metadata.updated_at

    conversation.add_user("Hello Atlas")

    assert (
            conversation.metadata.updated_at
            >= original_updated_at
    )


def test_adding_assistant_message_updates_metadata() -> None:
    conversation = Conversation()
    original_updated_at = conversation.metadata.updated_at

    conversation.add_assistant("Hello")

    assert (
            conversation.metadata.updated_at
            >= original_updated_at
    )


def test_adding_system_message_updates_metadata() -> None:
    conversation = Conversation()
    original_updated_at = conversation.metadata.updated_at

    conversation.add_system("You are Atlas.")

    assert (
            conversation.metadata.updated_at
            >= original_updated_at
    )


def test_adding_tool_message_updates_metadata() -> None:
    conversation = Conversation()
    original_updated_at = conversation.metadata.updated_at

    conversation.add_tool("Tool result")

    assert (
            conversation.metadata.updated_at
            >= original_updated_at
    )


def test_renaming_through_conversation_metadata() -> None:
    conversation = Conversation()

    conversation.metadata.rename("Market Research")

    assert conversation.metadata.title == "Market Research"


def test_renaming_does_not_change_messages() -> None:
    conversation = Conversation()
    message = conversation.add_user("Analyze BTC")

    conversation.metadata.rename("BTC Analysis")

    assert conversation.all() == (message,)
    assert len(conversation) == 1


def test_message_addition_does_not_change_title() -> None:
    conversation = Conversation(
        metadata=ConversationMetadata(
            title="Crypto Research"
        )
    )

    conversation.add_user("Analyze ETH")

    assert conversation.metadata.title == "Crypto Research"