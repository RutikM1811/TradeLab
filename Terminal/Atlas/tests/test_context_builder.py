from app.memory.conversation import Conversation
from app.prompts.context_builder import ContextBuilder
from app.types.message import Message, MessageRole


def test_empty_conversation_returns_empty_context() -> None:
    conversation = Conversation()
    builder = ContextBuilder()

    assert builder.build(conversation) == ""


def test_builds_system_message() -> None:
    conversation = Conversation()
    conversation.add_system("You are Atlas.")

    result = ContextBuilder().build(conversation)

    assert result == "System: You are Atlas."


def test_builds_user_message() -> None:
    conversation = Conversation()
    conversation.add_user("Analyze BTC")

    result = ContextBuilder().build(conversation)

    assert result == "User: Analyze BTC"


def test_builds_assistant_message() -> None:
    conversation = Conversation()
    conversation.add_assistant("BTC is trending upward.")

    result = ContextBuilder().build(conversation)

    assert result == "Assistant: BTC is trending upward."


def test_builds_tool_message() -> None:
    conversation = Conversation()
    conversation.add_tool("BTCUSDT price: 65000")

    result = ContextBuilder().build(conversation)

    assert result == "Tool: BTCUSDT price: 65000"


def test_preserves_message_order() -> None:
    conversation = Conversation()
    conversation.add_system("You are Atlas.")
    conversation.add_user("Analyze BTC")
    conversation.add_assistant("Analyzing BTC.")

    result = ContextBuilder().build(conversation)

    assert result == (
        "System: You are Atlas.\n"
        "User: Analyze BTC\n"
        "Assistant: Analyzing BTC."
    )


def test_builds_multi_turn_conversation() -> None:
    conversation = Conversation()
    conversation.add_user("Analyze BTC")
    conversation.add_assistant("BTC is bullish.")
    conversation.add_user("What about ETH?")
    conversation.add_assistant("ETH is also bullish.")

    result = ContextBuilder().build(conversation)

    assert result == (
        "User: Analyze BTC\n"
        "Assistant: BTC is bullish.\n"
        "User: What about ETH?\n"
        "Assistant: ETH is also bullish."
    )


def test_preserves_multiline_content() -> None:
    conversation = Conversation()
    conversation.add_user(
        "Analyze:\nBTC\nETH"
    )

    result = ContextBuilder().build(conversation)

    assert result == "User: Analyze:\nBTC\nETH"


def test_preserves_unicode_content() -> None:
    conversation = Conversation()
    conversation.add_user("नमस्कार Atlas 🚀")

    result = ContextBuilder().build(conversation)

    assert result == "User: नमस्कार Atlas 🚀"


def test_message_metadata_is_not_included() -> None:
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

    result = ContextBuilder().build(conversation)

    assert result == "Tool: Price fetched."
    assert "BTCUSDT" not in result
    assert "65000" not in result


def test_build_does_not_modify_conversation() -> None:
    conversation = Conversation()
    message = conversation.add_user("Hello Atlas")

    ContextBuilder().build(conversation)

    assert conversation.all() == (message,)
    assert len(conversation) == 1


def test_repeated_builds_are_deterministic() -> None:
    conversation = Conversation()
    conversation.add_user("Analyze BTC")
    conversation.add_assistant("BTC is bullish.")

    builder = ContextBuilder()

    first = builder.build(conversation)
    second = builder.build(conversation)

    assert first == second


def test_multiple_messages_are_separated_by_single_newline() -> None:
    conversation = Conversation()
    conversation.add_user("First")
    conversation.add_assistant("Second")

    result = ContextBuilder().build(conversation)

    assert result == "User: First\nAssistant: Second"


def test_same_role_messages_are_preserved() -> None:
    conversation = Conversation()
    conversation.add_user("First question")
    conversation.add_user("Second question")

    result = ContextBuilder().build(conversation)

    assert result == (
        "User: First question\n"
        "User: Second question"
    )


def test_all_supported_roles_build_together() -> None:
    conversation = Conversation()
    conversation.add_system("You are Atlas.")
    conversation.add_user("Get BTC price.")
    conversation.add_tool("BTC price is 65000.")
    conversation.add_assistant("BTC is trading at 65000.")

    result = ContextBuilder().build(conversation)

    assert result == (
        "System: You are Atlas.\n"
        "User: Get BTC price.\n"
        "Tool: BTC price is 65000.\n"
        "Assistant: BTC is trading at 65000."
    )