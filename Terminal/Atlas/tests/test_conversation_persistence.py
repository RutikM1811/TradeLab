from pathlib import Path
from uuid import uuid4

import pytest

from app.memory.conversation_manager import ConversationManager
from app.storage.local.json_conversation_storage import (
    JsonConversationStorage,
)


def create_manager(
        tmp_path: Path,
) -> ConversationManager:
    storage = JsonConversationStorage(tmp_path)

    return ConversationManager(
        storage=storage,
    )


def test_save_persists_managed_conversation(
        tmp_path: Path,
) -> None:
    manager = create_manager(tmp_path)

    conversation_id, conversation = manager.create()
    conversation.add_user("Hello Atlas")

    manager.save(conversation_id)

    storage = JsonConversationStorage(tmp_path)

    assert storage.contains(conversation_id)


def test_save_preserves_conversation_history(
        tmp_path: Path,
) -> None:
    manager = create_manager(tmp_path)

    conversation_id, conversation = manager.create()

    conversation.add_user("Analyze BTC")
    conversation.add_assistant("BTC is bullish.")

    manager.save(conversation_id)

    restored = JsonConversationStorage(
        tmp_path
    ).load(conversation_id)

    assert [message.content for message in restored.all()] == [
        "Analyze BTC",
        "BTC is bullish.",
    ]


def test_save_overwrites_persisted_conversation(
        tmp_path: Path,
) -> None:
    manager = create_manager(tmp_path)

    conversation_id, conversation = manager.create()

    conversation.add_user("First message")
    manager.save(conversation_id)

    conversation.add_assistant("Second message")
    manager.save(conversation_id)

    restored = JsonConversationStorage(
        tmp_path
    ).load(conversation_id)

    assert len(restored) == 2
    assert restored.last() is not None
    assert restored.last().content == "Second message"


def test_load_restores_conversation_into_memory(
        tmp_path: Path,
) -> None:
    first_manager = create_manager(tmp_path)

    conversation_id, conversation = (
        first_manager.create()
    )

    conversation.add_user("Persistent message")
    first_manager.save(conversation_id)

    second_manager = create_manager(tmp_path)

    restored = second_manager.load(conversation_id)

    assert second_manager.contains(conversation_id)
    assert second_manager.get(conversation_id) is restored


def test_load_restores_saved_history(
        tmp_path: Path,
) -> None:
    first_manager = create_manager(tmp_path)

    conversation_id, conversation = (
        first_manager.create()
    )

    conversation.add_user("Question")
    conversation.add_assistant("Answer")

    first_manager.save(conversation_id)

    second_manager = create_manager(tmp_path)

    restored = second_manager.load(conversation_id)

    assert [message.content for message in restored.all()] == [
        "Question",
        "Answer",
    ]


def test_restore_all_restores_multiple_conversations(
        tmp_path: Path,
) -> None:
    first_manager = create_manager(tmp_path)

    first_id, first = first_manager.create()
    second_id, second = first_manager.create()

    first.add_user("First conversation")
    second.add_user("Second conversation")

    first_manager.save(first_id)
    first_manager.save(second_id)

    second_manager = create_manager(tmp_path)

    restored_count = second_manager.restore_all()

    assert restored_count == 2
    assert second_manager.contains(first_id)
    assert second_manager.contains(second_id)
    assert len(second_manager) == 2


def test_restore_all_restores_conversation_content(
        tmp_path: Path,
) -> None:
    first_manager = create_manager(tmp_path)

    conversation_id, conversation = (
        first_manager.create()
    )

    conversation.add_user("Restart-safe message")
    first_manager.save(conversation_id)

    second_manager = create_manager(tmp_path)

    second_manager.restore_all()

    restored = second_manager.get(
        conversation_id
    )

    assert restored.last() is not None
    assert restored.last().content == (
        "Restart-safe message"
    )


def test_restore_all_on_empty_storage_returns_zero(
        tmp_path: Path,
) -> None:
    manager = create_manager(tmp_path)

    restored_count = manager.restore_all()

    assert restored_count == 0
    assert len(manager) == 0


def test_delete_removes_persisted_conversation(
        tmp_path: Path,
) -> None:
    manager = create_manager(tmp_path)

    conversation_id, _ = manager.create()

    manager.save(conversation_id)
    manager.delete(conversation_id)

    storage = JsonConversationStorage(tmp_path)

    assert storage.contains(conversation_id) is False
    assert manager.contains(conversation_id) is False


def test_delete_unsaved_conversation_still_works(
        tmp_path: Path,
) -> None:
    manager = create_manager(tmp_path)

    conversation_id, _ = manager.create()

    manager.delete(conversation_id)

    assert manager.contains(conversation_id) is False


def test_save_requires_configured_storage() -> None:
    manager = ConversationManager()

    conversation_id, _ = manager.create()

    with pytest.raises(
            RuntimeError,
            match="storage is not configured",
    ):
        manager.save(conversation_id)


def test_load_requires_configured_storage() -> None:
    manager = ConversationManager()

    with pytest.raises(
            RuntimeError,
            match="storage is not configured",
    ):
        manager.load(uuid4())


def test_restore_all_requires_configured_storage() -> None:
    manager = ConversationManager()

    with pytest.raises(
            RuntimeError,
            match="storage is not configured",
    ):
        manager.restore_all()


def test_simulated_restart_preserves_session_isolation(
        tmp_path: Path,
) -> None:
    first_manager = create_manager(tmp_path)

    first_id, first = first_manager.create()
    second_id, second = first_manager.create()

    first.add_user("Private first history")
    second.add_user("Private second history")

    first_manager.save(first_id)
    first_manager.save(second_id)

    restarted_manager = create_manager(tmp_path)

    restarted_manager.restore_all()

    restored_first = restarted_manager.get(first_id)
    restored_second = restarted_manager.get(second_id)

    assert restored_first.last() is not None
    assert restored_second.last() is not None

    assert restored_first.last().content == (
        "Private first history"
    )
    assert restored_second.last().content == (
        "Private second history"
    )


def test_in_memory_clear_does_not_delete_persisted_data(
        tmp_path: Path,
) -> None:
    manager = create_manager(tmp_path)

    conversation_id, conversation = manager.create()

    conversation.add_user("Keep this on disk")
    manager.save(conversation_id)

    manager.clear()

    assert len(manager) == 0

    restored_count = manager.restore_all()

    assert restored_count == 1
    assert manager.contains(conversation_id)