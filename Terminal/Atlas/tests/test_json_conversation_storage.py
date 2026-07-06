import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.memory.conversation import Conversation
from app.storage.local.json_conversation_storage import (
    JsonConversationStorage,
)
from app.types.message import Message, MessageRole


def test_storage_creates_directory(
        tmp_path: Path,
) -> None:
    directory = tmp_path / "conversations"

    JsonConversationStorage(directory)

    assert directory.exists()
    assert directory.is_dir()


def test_new_storage_is_empty(
        tmp_path: Path,
) -> None:
    storage = JsonConversationStorage(tmp_path)

    assert storage.all_ids() == ()


def test_save_creates_json_file(
        tmp_path: Path,
) -> None:
    storage = JsonConversationStorage(tmp_path)
    conversation_id = uuid4()

    storage.save(
        conversation_id,
        Conversation(),
    )

    path = tmp_path / f"{conversation_id}.json"

    assert path.exists()
    assert path.is_file()


def test_contains_returns_true_after_save(
        tmp_path: Path,
) -> None:
    storage = JsonConversationStorage(tmp_path)
    conversation_id = uuid4()

    storage.save(
        conversation_id,
        Conversation(),
    )

    assert storage.contains(conversation_id) is True


def test_contains_returns_false_for_missing_conversation(
        tmp_path: Path,
) -> None:
    storage = JsonConversationStorage(tmp_path)

    assert storage.contains(uuid4()) is False


def test_load_restores_saved_messages(
        tmp_path: Path,
) -> None:
    storage = JsonConversationStorage(tmp_path)
    conversation_id = uuid4()

    conversation = Conversation()
    conversation.add_user("Analyze BTC")
    conversation.add_assistant("BTC is bullish.")

    storage.save(
        conversation_id,
        conversation,
    )

    restored = storage.load(conversation_id)

    assert [message.content for message in restored.all()] == [
        "Analyze BTC",
        "BTC is bullish.",
    ]


def test_load_preserves_message_roles(
        tmp_path: Path,
) -> None:
    storage = JsonConversationStorage(tmp_path)
    conversation_id = uuid4()

    conversation = Conversation()
    conversation.add_system("You are Atlas.")
    conversation.add_user("Hello")
    conversation.add_tool("Tool result")
    conversation.add_assistant("Response")

    storage.save(
        conversation_id,
        conversation,
    )

    restored = storage.load(conversation_id)

    assert [message.role for message in restored.all()] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]


def test_load_preserves_message_metadata(
        tmp_path: Path,
) -> None:
    storage = JsonConversationStorage(tmp_path)
    conversation_id = uuid4()

    conversation = Conversation()
    conversation.add(
        Message(
            role=MessageRole.TOOL,
            content="Price fetched.",
            metadata={
                "symbol": "BTCUSDT",
                "price": 65000,
            },
        )
    )

    storage.save(
        conversation_id,
        conversation,
    )

    restored = storage.load(conversation_id)

    assert restored.all()[0].metadata == {
        "symbol": "BTCUSDT",
        "price": 65000,
    }


def test_save_overwrites_existing_conversation(
        tmp_path: Path,
) -> None:
    storage = JsonConversationStorage(tmp_path)
    conversation_id = uuid4()

    conversation = Conversation()
    conversation.add_user("First version")

    storage.save(
        conversation_id,
        conversation,
    )

    conversation.add_assistant("Updated version")

    storage.save(
        conversation_id,
        conversation,
    )

    restored = storage.load(conversation_id)

    assert len(restored) == 2
    assert restored.last() is not None
    assert restored.last().content == "Updated version"


def test_all_ids_returns_saved_conversation_ids(
        tmp_path: Path,
) -> None:
    storage = JsonConversationStorage(tmp_path)

    first_id = uuid4()
    second_id = uuid4()

    storage.save(first_id, Conversation())
    storage.save(second_id, Conversation())

    assert set(storage.all_ids()) == {
        first_id,
        second_id,
    }


def test_all_ids_ignores_non_json_files(
        tmp_path: Path,
) -> None:
    storage = JsonConversationStorage(tmp_path)
    conversation_id = uuid4()

    storage.save(
        conversation_id,
        Conversation(),
    )

    (tmp_path / "notes.txt").write_text(
        "ignore me",
        encoding="utf-8",
    )

    assert storage.all_ids() == (
        conversation_id,
    )


def test_delete_removes_saved_conversation(
        tmp_path: Path,
) -> None:
    storage = JsonConversationStorage(tmp_path)
    conversation_id = uuid4()

    storage.save(
        conversation_id,
        Conversation(),
    )

    storage.delete(conversation_id)

    assert storage.contains(conversation_id) is False


def test_delete_raises_for_missing_conversation(
        tmp_path: Path,
) -> None:
    storage = JsonConversationStorage(tmp_path)

    with pytest.raises(
            KeyError,
            match="was not found",
    ):
        storage.delete(uuid4())


def test_load_raises_for_missing_conversation(
        tmp_path: Path,
) -> None:
    storage = JsonConversationStorage(tmp_path)

    with pytest.raises(
            KeyError,
            match="was not found",
    ):
        storage.load(uuid4())


def test_clear_removes_all_saved_conversations(
        tmp_path: Path,
) -> None:
    storage = JsonConversationStorage(tmp_path)

    storage.save(uuid4(), Conversation())
    storage.save(uuid4(), Conversation())
    storage.save(uuid4(), Conversation())

    storage.clear()

    assert storage.all_ids() == ()


def test_clear_preserves_unrelated_files(
        tmp_path: Path,
) -> None:
    storage = JsonConversationStorage(tmp_path)

    storage.save(
        uuid4(),
        Conversation(),
    )

    unrelated = tmp_path / "notes.txt"
    unrelated.write_text(
        "keep me",
        encoding="utf-8",
    )

    storage.clear()

    assert unrelated.exists()
    assert unrelated.read_text(
        encoding="utf-8"
    ) == "keep me"


def test_load_raises_for_corrupted_json(
        tmp_path: Path,
) -> None:
    storage = JsonConversationStorage(tmp_path)
    conversation_id = uuid4()

    path = tmp_path / f"{conversation_id}.json"

    path.write_text(
        "{invalid json",
        encoding="utf-8",
    )

    with pytest.raises(json.JSONDecodeError):
        storage.load(conversation_id)


def test_load_rejects_mismatched_stored_id(
        tmp_path: Path,
) -> None:
    storage = JsonConversationStorage(tmp_path)

    requested_id = uuid4()
    stored_id = uuid4()

    path = tmp_path / f"{requested_id}.json"

    path.write_text(
        json.dumps(
            {
                "id": str(stored_id),
                "messages": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
            ValueError,
            match="does not match requested ID",
    ):
        storage.load(requested_id)