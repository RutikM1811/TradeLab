from datetime import UTC

import pytest

from app.memory.conversation_metadata import (
    ConversationMetadata,
)


def test_default_title_is_new_conversation() -> None:
    metadata = ConversationMetadata()

    assert metadata.title == "New Conversation"


def test_created_at_is_utc() -> None:
    metadata = ConversationMetadata()

    assert metadata.created_at.tzinfo is UTC


def test_updated_at_is_utc() -> None:
    metadata = ConversationMetadata()

    assert metadata.updated_at.tzinfo is UTC


def test_created_and_updated_times_are_initialized() -> None:
    metadata = ConversationMetadata()

    assert metadata.created_at is not None
    assert metadata.updated_at is not None


def test_custom_title_is_supported() -> None:
    metadata = ConversationMetadata(
        title="BTC Analysis"
    )

    assert metadata.title == "BTC Analysis"


def test_rename_changes_title() -> None:
    metadata = ConversationMetadata()

    metadata.rename("ETH Analysis")

    assert metadata.title == "ETH Analysis"


def test_rename_strips_surrounding_whitespace() -> None:
    metadata = ConversationMetadata()

    metadata.rename("   Market Research   ")

    assert metadata.title == "Market Research"


def test_rename_rejects_empty_title() -> None:
    metadata = ConversationMetadata()

    with pytest.raises(
            ValueError,
            match="Conversation title cannot be empty",
    ):
        metadata.rename("")


def test_rename_rejects_whitespace_only_title() -> None:
    metadata = ConversationMetadata()

    with pytest.raises(
            ValueError,
            match="Conversation title cannot be empty",
    ):
        metadata.rename("   ")


def test_touch_updates_updated_at() -> None:
    metadata = ConversationMetadata()
    original_updated_at = metadata.updated_at

    metadata.touch()

    assert metadata.updated_at >= original_updated_at


def test_rename_updates_updated_at() -> None:
    metadata = ConversationMetadata()
    original_updated_at = metadata.updated_at

    metadata.rename("New Title")

    assert metadata.updated_at >= original_updated_at


def test_rename_does_not_change_created_at() -> None:
    metadata = ConversationMetadata()
    original_created_at = metadata.created_at

    metadata.rename("Persistent Title")

    assert metadata.created_at == original_created_at