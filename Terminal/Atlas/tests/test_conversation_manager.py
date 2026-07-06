from uuid import UUID, uuid4

import pytest

from app.memory.conversation import Conversation
from app.memory.conversation_manager import ConversationManager


def test_new_manager_is_empty() -> None:
    manager = ConversationManager()

    assert len(manager) == 0
    assert manager.all() == ()


def test_create_returns_uuid_and_conversation() -> None:
    manager = ConversationManager()

    conversation_id, conversation = manager.create()

    assert isinstance(conversation_id, UUID)
    assert isinstance(conversation, Conversation)


def test_create_stores_conversation() -> None:
    manager = ConversationManager()

    conversation_id, conversation = manager.create()

    assert manager.contains(conversation_id)
    assert manager.get(conversation_id) is conversation
    assert len(manager) == 1


def test_multiple_conversations_have_unique_ids() -> None:
    manager = ConversationManager()

    first_id, _ = manager.create()
    second_id, _ = manager.create()

    assert first_id != second_id


def test_get_returns_same_conversation_instance() -> None:
    manager = ConversationManager()

    conversation_id, conversation = manager.create()

    resolved = manager.get(conversation_id)

    assert resolved is conversation


def test_get_raises_for_missing_conversation() -> None:
    manager = ConversationManager()
    missing_id = uuid4()

    with pytest.raises(
            KeyError,
            match="was not found",
    ):
        manager.get(missing_id)


def test_contains_returns_false_for_missing_conversation() -> None:
    manager = ConversationManager()

    assert manager.contains(uuid4()) is False


def test_all_returns_all_conversations() -> None:
    manager = ConversationManager()

    first_id, first = manager.create()
    second_id, second = manager.create()

    assert manager.all() == (
        (first_id, first),
        (second_id, second),
    )


def test_all_preserves_creation_order() -> None:
    manager = ConversationManager()

    first_id, _ = manager.create()
    second_id, _ = manager.create()
    third_id, _ = manager.create()

    ids = tuple(
        conversation_id
        for conversation_id, _ in manager.all()
    )

    assert ids == (
        first_id,
        second_id,
        third_id,
    )


def test_all_returns_immutable_tuple() -> None:
    manager = ConversationManager()
    manager.create()

    assert isinstance(manager.all(), tuple)


def test_conversations_are_isolated() -> None:
    manager = ConversationManager()

    _, first = manager.create()
    _, second = manager.create()

    first.add_user("Message in first conversation")

    assert len(first) == 1
    assert len(second) == 0
    assert second.all() == ()


def test_delete_removes_conversation() -> None:
    manager = ConversationManager()

    conversation_id, _ = manager.create()

    manager.delete(conversation_id)

    assert manager.contains(conversation_id) is False
    assert len(manager) == 0


def test_delete_does_not_remove_other_conversations() -> None:
    manager = ConversationManager()

    first_id, _ = manager.create()
    second_id, second = manager.create()

    manager.delete(first_id)

    assert manager.contains(first_id) is False
    assert manager.contains(second_id) is True
    assert manager.get(second_id) is second


def test_delete_raises_for_missing_conversation() -> None:
    manager = ConversationManager()

    with pytest.raises(
            KeyError,
            match="was not found",
    ):
        manager.delete(uuid4())


def test_clear_removes_all_conversations() -> None:
    manager = ConversationManager()

    manager.create()
    manager.create()
    manager.create()

    manager.clear()

    assert len(manager) == 0
    assert manager.all() == ()