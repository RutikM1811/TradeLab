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