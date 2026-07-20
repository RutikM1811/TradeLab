from uuid import UUID, uuid4

import pytest

from app.memory.conversation import Conversation
from app.memory.conversation_manager import ConversationManager
from unittest.mock import Mock

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
def test_clear_on_empty_manager_does_not_fail() -> None:
    manager = ConversationManager()
    manager.clear()
    assert len(manager) == 0
    assert manager.all() == ()
def test_len_increases_after_each_create() -> None:
    manager = ConversationManager()

    assert len(manager) == 0

    manager.create()
    assert len(manager) == 1

    manager.create()
    assert len(manager) == 2

    manager.create()
    assert len(manager) == 3


def test_len_decreases_after_delete() -> None:
    manager = ConversationManager()

    first, _ = manager.create()
    second, _ = manager.create()

    assert len(manager) == 2

    manager.delete(first)
    assert len(manager) == 1

    manager.delete(second)
    assert len(manager) == 0


def test_contains_returns_true_for_existing_conversation() -> None:
    manager = ConversationManager()

    conversation_id, _ = manager.create()

    assert manager.contains(conversation_id)


def test_get_returns_same_instance_multiple_times() -> None:
    manager = ConversationManager()

    conversation_id, conversation = manager.create()

    assert manager.get(conversation_id) is conversation
    assert manager.get(conversation_id) is conversation
    assert manager.get(conversation_id) is conversation


def test_delete_twice_raises_key_error() -> None:
    manager = ConversationManager()

    conversation_id, _ = manager.create()

    manager.delete(conversation_id)

    with pytest.raises(KeyError):
        manager.delete(conversation_id)


def test_save_without_storage_raises_runtime_error() -> None:
    manager = ConversationManager()

    conversation_id, _ = manager.create()

    with pytest.raises(RuntimeError, match="Conversation storage is not configured"):
        manager.save(conversation_id)


def test_load_without_storage_raises_runtime_error() -> None:
    manager = ConversationManager()

    with pytest.raises(RuntimeError, match="Conversation storage is not configured"):
        manager.load(uuid4())


def test_restore_all_without_storage_raises_runtime_error() -> None:
    manager = ConversationManager()

    with pytest.raises(RuntimeError, match="Conversation storage is not configured"):
        manager.restore_all()


def test_save_calls_storage_save() -> None:
    storage = Mock()

    manager = ConversationManager(storage)

    conversation_id, conversation = manager.create()

    manager.save(conversation_id)

    storage.save.assert_called_once_with(
        conversation_id,
        conversation,
    )


def test_load_calls_storage_load() -> None:
    storage = Mock()

    conversation = Conversation()

    storage.load.return_value = conversation

    manager = ConversationManager(storage)

    conversation_id = uuid4()

    loaded = manager.load(conversation_id)

    assert loaded is conversation

    storage.load.assert_called_once_with(conversation_id)


def test_restore_all_returns_zero_when_storage_empty() -> None:
    storage = Mock()

    storage.all_ids.return_value = ()

    manager = ConversationManager(storage)

    restored = manager.restore_all()

    assert restored == 0


def test_restore_all_loads_every_conversation() -> None:
    storage = Mock()

    ids = [uuid4(), uuid4(), uuid4()]

    storage.all_ids.return_value = ids
    storage.load.return_value = Conversation()

    manager = ConversationManager(storage)

    restored = manager.restore_all()

    assert restored == 3
    assert storage.load.call_count == 3


def test_delete_calls_storage_delete_when_present() -> None:
    storage = Mock()

    storage.contains.return_value = True

    manager = ConversationManager(storage)

    conversation_id, _ = manager.create()

    manager.delete(conversation_id)

    storage.delete.assert_called_once_with(conversation_id)


def test_delete_does_not_call_storage_delete_when_not_present() -> None:
    storage = Mock()

    storage.contains.return_value = False

    manager = ConversationManager(storage)

    conversation_id, _ = manager.create()

    manager.delete(conversation_id)

    storage.delete.assert_not_called()


def test_load_stores_conversation_in_manager() -> None:
    storage = Mock()

    conversation = Conversation()

    storage.load.return_value = conversation

    manager = ConversationManager(storage)

    conversation_id = uuid4()

    manager.load(conversation_id)

    assert manager.contains(conversation_id)
    assert manager.get(conversation_id) is conversation


def test_restore_all_stores_loaded_conversations() -> None:
    storage = Mock()

    ids = [uuid4(), uuid4()]

    conversations = [Conversation(), Conversation()]

    storage.all_ids.return_value = ids
    storage.load.side_effect = conversations

    manager = ConversationManager(storage)

    manager.restore_all()

    assert manager.get(ids[0]) is conversations[0]
    assert manager.get(ids[1]) is conversations[1]