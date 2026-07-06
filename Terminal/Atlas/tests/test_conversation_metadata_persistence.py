from pathlib import Path
from uuid import uuid4

from app.memory.conversation import Conversation
from app.memory.conversation_metadata import ConversationMetadata
from app.memory.conversation_serializer import ConversationSerializer
from app.storage.local.json_conversation_storage import (
    JsonConversationStorage,
)


def test_serialize_includes_conversation_metadata() -> None:
    serializer = ConversationSerializer()
    conversation = Conversation()

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    assert "metadata" in data


def test_serialize_preserves_title() -> None:
    serializer = ConversationSerializer()

    conversation = Conversation(
        metadata=ConversationMetadata(
            title="BTC Analysis"
        )
    )

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    assert data["metadata"]["title"] == "BTC Analysis"


def test_serialize_preserves_created_at() -> None:
    serializer = ConversationSerializer()
    conversation = Conversation()

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    assert data["metadata"]["created_at"] == (
        conversation.metadata.created_at.isoformat()
    )


def test_serialize_preserves_updated_at() -> None:
    serializer = ConversationSerializer()
    conversation = Conversation()

    conversation.add_user("Hello")

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    assert data["metadata"]["updated_at"] == (
        conversation.metadata.updated_at.isoformat()
    )


def test_round_trip_preserves_title() -> None:
    serializer = ConversationSerializer()

    conversation = Conversation(
        metadata=ConversationMetadata(
            title="Market Research"
        )
    )

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    _, restored = serializer.deserialize(data)

    assert restored.metadata.title == "Market Research"


def test_round_trip_preserves_metadata_timestamps() -> None:
    serializer = ConversationSerializer()
    conversation = Conversation()

    original_created_at = conversation.metadata.created_at
    original_updated_at = conversation.metadata.updated_at

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    _, restored = serializer.deserialize(data)

    assert restored.metadata.created_at == original_created_at
    assert restored.metadata.updated_at == original_updated_at


def test_round_trip_preserves_metadata_and_messages() -> None:
    serializer = ConversationSerializer()

    conversation = Conversation(
        metadata=ConversationMetadata(
            title="ETH Analysis"
        )
    )

    conversation.add_user("Analyze ETH")
    conversation.add_assistant("ETH is bullish.")

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    _, restored = serializer.deserialize(data)

    assert restored.metadata.title == "ETH Analysis"

    assert [message.content for message in restored.all()] == [
        "Analyze ETH",
        "ETH is bullish.",
    ]


def test_deserialize_supports_legacy_data_without_metadata() -> None:
    serializer = ConversationSerializer()

    conversation_id = uuid4()

    restored_id, restored = serializer.deserialize(
        {
            "id": str(conversation_id),
            "messages": [],
        }
    )

    assert restored_id == conversation_id
    assert restored.metadata.title == "New Conversation"


def test_legacy_conversation_receives_valid_metadata() -> None:
    serializer = ConversationSerializer()

    _, restored = serializer.deserialize(
        {
            "id": str(uuid4()),
            "messages": [],
        }
    )

    assert restored.metadata.created_at is not None
    assert restored.metadata.updated_at is not None


def test_json_storage_preserves_title(
        tmp_path: Path,
) -> None:
    storage = JsonConversationStorage(tmp_path)

    conversation_id = uuid4()

    conversation = Conversation(
        metadata=ConversationMetadata(
            title="Persistent BTC Chat"
        )
    )

    storage.save(
        conversation_id,
        conversation,
    )

    restored = storage.load(conversation_id)

    assert restored.metadata.title == "Persistent BTC Chat"


def test_json_storage_preserves_metadata_timestamps(
        tmp_path: Path,
) -> None:
    storage = JsonConversationStorage(tmp_path)

    conversation_id = uuid4()
    conversation = Conversation()

    created_at = conversation.metadata.created_at
    updated_at = conversation.metadata.updated_at

    storage.save(
        conversation_id,
        conversation,
    )

    restored = storage.load(conversation_id)

    assert restored.metadata.created_at == created_at
    assert restored.metadata.updated_at == updated_at


def test_metadata_survives_simulated_restart(
        tmp_path: Path,
) -> None:
    first_storage = JsonConversationStorage(tmp_path)

    conversation_id = uuid4()

    conversation = Conversation(
        metadata=ConversationMetadata(
            title="Restart Safe Chat"
        )
    )

    conversation.add_user("Persistent message")

    first_storage.save(
        conversation_id,
        conversation,
    )

    restarted_storage = JsonConversationStorage(tmp_path)

    restored = restarted_storage.load(
        conversation_id
    )

    assert restored.metadata.title == "Restart Safe Chat"
    assert restored.last() is not None
    assert restored.last().content == "Persistent message"