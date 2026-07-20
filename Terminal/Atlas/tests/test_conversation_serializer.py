from datetime import UTC
from uuid import UUID, uuid4

from app.memory.conversation import Conversation
from app.memory.conversation_serializer import ConversationSerializer
from app.types.message import Message, MessageRole


def test_serialize_returns_dictionary() -> None:
    serializer = ConversationSerializer()
    conversation_id = uuid4()

    result = serializer.serialize(
        conversation_id,
        Conversation(),
    )

    assert isinstance(result, dict)


def test_serialize_preserves_conversation_id() -> None:
    serializer = ConversationSerializer()
    conversation_id = uuid4()

    result = serializer.serialize(
        conversation_id,
        Conversation(),
    )

    assert result["id"] == str(conversation_id)


def test_serialize_empty_conversation() -> None:
    serializer = ConversationSerializer()

    result = serializer.serialize(
        uuid4(),
        Conversation(),
    )

    assert result["messages"] == []


def test_serialize_preserves_message_content() -> None:
    serializer = ConversationSerializer()
    conversation = Conversation()

    conversation.add_user("Analyze BTC")

    result = serializer.serialize(
        uuid4(),
        conversation,
    )

    assert result["messages"][0]["content"] == "Analyze BTC"


def test_serialize_preserves_message_role() -> None:
    serializer = ConversationSerializer()
    conversation = Conversation()

    conversation.add_assistant("BTC is bullish.")

    result = serializer.serialize(
        uuid4(),
        conversation,
    )

    assert result["messages"][0]["role"] == "assistant"


def test_serialize_preserves_message_id() -> None:
    serializer = ConversationSerializer()
    conversation = Conversation()

    message = conversation.add_user("Hello")

    result = serializer.serialize(
        uuid4(),
        conversation,
    )

    assert result["messages"][0]["id"] == str(message.id)


def test_serialize_preserves_utc_timestamp() -> None:
    serializer = ConversationSerializer()
    conversation = Conversation()

    message = conversation.add_user("Hello")

    result = serializer.serialize(
        uuid4(),
        conversation,
    )

    assert result["messages"][0]["created_at"] == (
        message.created_at.isoformat()
    )


def test_serialize_preserves_metadata() -> None:
    serializer = ConversationSerializer()
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

    result = serializer.serialize(
        uuid4(),
        conversation,
    )

    assert result["messages"][0]["metadata"] == {
        "symbol": "BTCUSDT",
        "price": 65000,
    }


def test_deserialize_restores_conversation_id() -> None:
    serializer = ConversationSerializer()
    conversation_id = uuid4()

    restored_id, _ = serializer.deserialize(
        {
            "id": str(conversation_id),
            "messages": [],
        }
    )

    assert restored_id == conversation_id
    assert isinstance(restored_id, UUID)


def test_deserialize_restores_empty_conversation() -> None:
    serializer = ConversationSerializer()

    _, conversation = serializer.deserialize(
        {
            "id": str(uuid4()),
            "messages": [],
        }
    )

    assert len(conversation) == 0
    assert conversation.all() == ()


def test_round_trip_preserves_all_message_roles() -> None:
    serializer = ConversationSerializer()
    conversation_id = uuid4()
    conversation = Conversation()

    conversation.add_system("You are Atlas.")
    conversation.add_user("Analyze BTC")
    conversation.add_assistant("BTC is bullish.")
    conversation.add_tool("BTC price is 65000")

    data = serializer.serialize(
        conversation_id,
        conversation,
    )

    restored_id, restored = serializer.deserialize(data)

    assert restored_id == conversation_id
    assert [message.role for message in restored.all()] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]


def test_round_trip_preserves_message_order() -> None:
    serializer = ConversationSerializer()
    conversation = Conversation()

    conversation.add_user("First")
    conversation.add_assistant("Second")
    conversation.add_user("Third")

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    _, restored = serializer.deserialize(data)

    assert [message.content for message in restored.all()] == [
        "First",
        "Second",
        "Third",
    ]


def test_round_trip_preserves_message_identity() -> None:
    serializer = ConversationSerializer()
    conversation = Conversation()

    original = conversation.add_user("Hello")

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    _, restored = serializer.deserialize(data)

    restored_message = restored.all()[0]

    assert restored_message.id == original.id
    assert restored_message.created_at == original.created_at


def test_round_trip_preserves_metadata() -> None:
    serializer = ConversationSerializer()
    conversation = Conversation()

    conversation.add(
        Message(
            role=MessageRole.TOOL,
            content="Market data loaded.",
            metadata={
                "symbol": "ETHUSDT",
                "timeframe": "4h",
            },
        )
    )

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    _, restored = serializer.deserialize(data)

    assert restored.all()[0].metadata == {
        "symbol": "ETHUSDT",
        "timeframe": "4h",
    }


def test_restored_timestamp_remains_timezone_aware() -> None:
    serializer = ConversationSerializer()
    conversation = Conversation()

    conversation.add_user("Hello")

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    _, restored = serializer.deserialize(data)

    assert restored.all()[0].created_at.tzinfo is UTC
from datetime import datetime


def test_serialize_includes_metadata_section() -> None:
    serializer = ConversationSerializer()

    data = serializer.serialize(
        uuid4(),
        Conversation(),
    )

    assert "metadata" in data


def test_serialize_preserves_title() -> None:
    serializer = ConversationSerializer()

    conversation = Conversation()
    conversation.metadata.rename("Trading Chat")

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    assert data["metadata"]["title"] == "Trading Chat"


def test_serialize_preserves_created_at() -> None:
    serializer = ConversationSerializer()

    conversation = Conversation()

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    assert (
            data["metadata"]["created_at"]
            == conversation.metadata.created_at.isoformat()
    )


def test_serialize_preserves_updated_at() -> None:
    serializer = ConversationSerializer()

    conversation = Conversation()

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    assert (
            data["metadata"]["updated_at"]
            == conversation.metadata.updated_at.isoformat()
    )


def test_deserialize_without_metadata_creates_default_metadata() -> None:
    serializer = ConversationSerializer()

    _, conversation = serializer.deserialize(
        {
            "id": str(uuid4()),
            "messages": [],
        }
    )

    assert conversation.metadata is not None


def test_round_trip_preserves_title() -> None:
    serializer = ConversationSerializer()

    conversation = Conversation()
    conversation.metadata.rename("Crypto")

    conversation_id = uuid4()

    data = serializer.serialize(
        conversation_id,
        conversation,
    )

    _, restored = serializer.deserialize(data)

    assert restored.metadata.title == "Crypto"


def test_round_trip_preserves_created_at() -> None:
    serializer = ConversationSerializer()

    conversation = Conversation()

    created = conversation.metadata.created_at

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    _, restored = serializer.deserialize(data)

    assert restored.metadata.created_at == created


def test_round_trip_preserves_updated_at() -> None:
    serializer = ConversationSerializer()

    conversation = Conversation()

    updated = conversation.metadata.updated_at

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    _, restored = serializer.deserialize(data)

    assert restored.metadata.updated_at == updated


def test_deserialize_restores_system_message() -> None:
    serializer = ConversationSerializer()

    data = {
        "id": str(uuid4()),
        "metadata": {
            "title": "New Conversation",
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        },
        "messages": [
            {
                "id": str(uuid4()),
                "role": "system",
                "content": "You are Atlas",
                "created_at": datetime.now(UTC).isoformat(),
                "metadata": {},
            }
        ],
    }

    _, conversation = serializer.deserialize(data)

    assert conversation.last().role is MessageRole.SYSTEM


def test_deserialize_restores_tool_message() -> None:
    serializer = ConversationSerializer()

    data = {
        "id": str(uuid4()),
        "metadata": {
            "title": "New Conversation",
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        },
        "messages": [
            {
                "id": str(uuid4()),
                "role": "tool",
                "content": "Output",
                "created_at": datetime.now(UTC).isoformat(),
                "metadata": {},
            }
        ],
    }

    _, conversation = serializer.deserialize(data)

    assert conversation.last().role is MessageRole.TOOL


def test_deserialize_defaults_missing_message_metadata() -> None:
    serializer = ConversationSerializer()

    message_id = uuid4()

    data = {
        "id": str(uuid4()),
        "messages": [
            {
                "id": str(message_id),
                "role": "user",
                "content": "Hello",
                "created_at": datetime.now(UTC).isoformat(),
            }
        ],
    }

    _, conversation = serializer.deserialize(data)

    assert conversation.last().metadata == {}


def test_round_trip_preserves_unicode() -> None:
    serializer = ConversationSerializer()

    conversation = Conversation()
    conversation.add_user("नमस्कार 🚀")

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    _, restored = serializer.deserialize(data)

    assert restored.last().content == "नमस्कार 🚀"


def test_round_trip_preserves_multiline_message() -> None:
    serializer = ConversationSerializer()

    conversation = Conversation()
    conversation.add_user("One\nTwo\nThree")

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    _, restored = serializer.deserialize(data)

    assert restored.last().content == "One\nTwo\nThree"


def test_round_trip_preserves_empty_metadata_dict() -> None:
    serializer = ConversationSerializer()

    conversation = Conversation()

    conversation.add(
        Message(
            role=MessageRole.USER,
            content="Hello",
            metadata={},
        )
    )

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    _, restored = serializer.deserialize(data)

    assert restored.last().metadata == {}


def test_serialize_message_count_matches_conversation() -> None:
    serializer = ConversationSerializer()

    conversation = Conversation()

    for i in range(7):
        conversation.add_user(str(i))

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    assert len(data["messages"]) == 7


def test_deserialize_message_count_matches() -> None:
    serializer = ConversationSerializer()

    conversation = Conversation()

    for i in range(7):
        conversation.add_user(str(i))

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    _, restored = serializer.deserialize(data)

    assert len(restored) == 7


def test_round_trip_preserves_first_message() -> None:
    serializer = ConversationSerializer()

    conversation = Conversation()

    conversation.add_user("First")
    conversation.add_user("Second")

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    _, restored = serializer.deserialize(data)

    assert restored.all()[0].content == "First"


def test_round_trip_preserves_last_message() -> None:
    serializer = ConversationSerializer()

    conversation = Conversation()

    conversation.add_user("First")
    conversation.add_assistant("Last")

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    _, restored = serializer.deserialize(data)

    assert restored.last().content == "Last"


def test_deserialize_returns_uuid_instance() -> None:
    serializer = ConversationSerializer()

    conversation_id = uuid4()

    restored_id, _ = serializer.deserialize(
        {
            "id": str(conversation_id),
            "messages": [],
        }
    )

    assert isinstance(restored_id, UUID)


def test_serialized_message_ids_are_unique() -> None:
    serializer = ConversationSerializer()

    conversation = Conversation()

    for i in range(5):
        conversation.add_user(str(i))

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    ids = [m["id"] for m in data["messages"]]

    assert len(ids) == len(set(ids))


def test_deserialize_preserves_message_id() -> None:
    serializer = ConversationSerializer()

    conversation = Conversation()

    original = conversation.add_user("Hello")

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    _, restored = serializer.deserialize(data)

    assert restored.last().id == original.id


def test_deserialize_preserves_role_order() -> None:
    serializer = ConversationSerializer()

    conversation = Conversation()

    conversation.add_system("S")
    conversation.add_user("U")
    conversation.add_tool("T")
    conversation.add_assistant("A")

    data = serializer.serialize(
        uuid4(),
        conversation,
    )

    _, restored = serializer.deserialize(data)

    assert [m.role for m in restored.all()] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]