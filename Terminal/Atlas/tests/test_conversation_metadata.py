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
from datetime import datetime, timedelta


def test_default_title_has_no_extra_spaces() -> None:
    metadata = ConversationMetadata()

    assert metadata.title == metadata.title.strip()


def test_custom_title_preserves_exact_value() -> None:
    metadata = ConversationMetadata(
        title="Atlas Chat"
    )

    assert metadata.title == "Atlas Chat"


def test_created_at_is_not_after_updated_at() -> None:
    metadata = ConversationMetadata()

    assert metadata.created_at <= metadata.updated_at


def test_touch_does_not_change_created_at() -> None:
    metadata = ConversationMetadata()

    created = metadata.created_at

    metadata.touch()

    assert metadata.created_at == created


def test_touch_can_be_called_multiple_times() -> None:
    metadata = ConversationMetadata()

    metadata.touch()
    first = metadata.updated_at

    metadata.touch()

    assert metadata.updated_at >= first


def test_rename_to_same_title_is_allowed() -> None:
    metadata = ConversationMetadata()

    metadata.rename("New Conversation")

    assert metadata.title == "New Conversation"


def test_rename_preserves_unicode() -> None:
    metadata = ConversationMetadata()

    metadata.rename("नमस्कार")

    assert metadata.title == "नमस्कार"


def test_rename_preserves_emoji() -> None:
    metadata = ConversationMetadata()

    metadata.rename("🚀 Atlas")

    assert metadata.title == "🚀 Atlas"


def test_rename_accepts_long_title() -> None:
    metadata = ConversationMetadata()

    title = "A" * 500

    metadata.rename(title)

    assert metadata.title == title


def test_rename_removes_tabs() -> None:
    metadata = ConversationMetadata()

    metadata.rename("\tAtlas\t")

    assert metadata.title == "Atlas"


def test_rename_removes_newlines() -> None:
    metadata = ConversationMetadata()

    metadata.rename("\nAtlas\n")

    assert metadata.title == "Atlas"


def test_multiple_renames_keep_latest_title() -> None:
    metadata = ConversationMetadata()

    metadata.rename("One")
    metadata.rename("Two")
    metadata.rename("Three")

    assert metadata.title == "Three"


def test_multiple_renames_do_not_change_created_at() -> None:
    metadata = ConversationMetadata()

    created = metadata.created_at

    metadata.rename("One")
    metadata.rename("Two")

    assert metadata.created_at == created


def test_touch_keeps_timezone_information() -> None:
    metadata = ConversationMetadata()

    metadata.touch()

    assert metadata.updated_at.tzinfo is UTC


def test_rename_keeps_timezone_information() -> None:
    metadata = ConversationMetadata()

    metadata.rename("Atlas")

    assert metadata.updated_at.tzinfo is UTC


def test_custom_created_at_is_supported() -> None:
    created = datetime.now(UTC) - timedelta(days=1)

    metadata = ConversationMetadata(
        created_at=created
    )

    assert metadata.created_at == created


def test_custom_updated_at_is_supported() -> None:
    updated = datetime.now(UTC) - timedelta(hours=1)

    metadata = ConversationMetadata(
        updated_at=updated
    )

    assert metadata.updated_at == updated


def test_touch_updates_time_forward() -> None:
    metadata = ConversationMetadata()

    before = metadata.updated_at

    metadata.touch()

    assert metadata.updated_at >= before


def test_rename_updates_time_forward() -> None:
    metadata = ConversationMetadata()

    before = metadata.updated_at

    metadata.rename("Atlas")

    assert metadata.updated_at >= before


def test_default_title_is_not_empty() -> None:
    metadata = ConversationMetadata()

    assert metadata.title != ""