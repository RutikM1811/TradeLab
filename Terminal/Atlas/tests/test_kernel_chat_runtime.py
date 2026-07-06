from uuid import UUID, uuid4

import pytest

from app.kernel.bootstrap import Kernel
from app.types.message import MessageRole


def test_kernel_creates_conversation() -> None:
    kernel = Kernel()
    kernel.boot()

    conversation_id, conversation = kernel.create_conversation()

    assert isinstance(conversation_id, UUID)
    assert len(conversation) == 0


def test_kernel_gets_created_conversation() -> None:
    kernel = Kernel()
    kernel.boot()

    conversation_id, conversation = kernel.create_conversation()

    assert kernel.get_conversation(conversation_id) is conversation


def test_kernel_created_conversations_have_unique_ids() -> None:
    kernel = Kernel()
    kernel.boot()

    first_id, _ = kernel.create_conversation()
    second_id, _ = kernel.create_conversation()

    assert first_id != second_id


@pytest.mark.anyio
async def test_kernel_chat_returns_successful_result() -> None:
    kernel = Kernel()
    kernel.boot()

    conversation_id, _ = kernel.create_conversation()

    result = await kernel.chat(
        conversation_id,
        "Hello Atlas",
    )

    assert result.success is True
    assert result.content == "Atlas development response."


@pytest.mark.anyio
async def test_kernel_chat_stores_user_message() -> None:
    kernel = Kernel()
    kernel.boot()

    conversation_id, conversation = kernel.create_conversation()

    await kernel.chat(
        conversation_id,
        "Analyze BTC",
    )

    assert conversation.all()[0].role is MessageRole.USER
    assert conversation.all()[0].content == "Analyze BTC"


@pytest.mark.anyio
async def test_kernel_chat_stores_assistant_response() -> None:
    kernel = Kernel()
    kernel.boot()

    conversation_id, conversation = kernel.create_conversation()

    await kernel.chat(
        conversation_id,
        "Hello",
    )

    assert conversation.all()[1].role is MessageRole.ASSISTANT
    assert (
            conversation.all()[1].content
            == "Atlas development response."
    )


@pytest.mark.anyio
async def test_kernel_chat_adds_two_messages_per_successful_turn() -> None:
    kernel = Kernel()
    kernel.boot()

    conversation_id, conversation = kernel.create_conversation()

    await kernel.chat(
        conversation_id,
        "Hello",
    )

    assert len(conversation) == 2


@pytest.mark.anyio
async def test_kernel_chat_preserves_multi_turn_history() -> None:
    kernel = Kernel()
    kernel.boot()

    conversation_id, conversation = kernel.create_conversation()

    await kernel.chat(
        conversation_id,
        "First question",
    )

    await kernel.chat(
        conversation_id,
        "Second question",
    )

    assert len(conversation) == 4

    assert [message.content for message in conversation.all()] == [
        "First question",
        "Atlas development response.",
        "Second question",
        "Atlas development response.",
    ]


@pytest.mark.anyio
async def test_kernel_chat_sessions_are_isolated() -> None:
    kernel = Kernel()
    kernel.boot()

    first_id, first = kernel.create_conversation()
    second_id, second = kernel.create_conversation()

    await kernel.chat(
        first_id,
        "First session",
    )

    await kernel.chat(
        second_id,
        "Second session",
    )

    assert first.all()[0].content == "First session"
    assert second.all()[0].content == "Second session"

    assert len(first) == 2
    assert len(second) == 2


@pytest.mark.anyio
async def test_kernel_chat_result_contains_atlas_metadata() -> None:
    kernel = Kernel()
    kernel.boot()

    conversation_id, _ = kernel.create_conversation()

    result = await kernel.chat(
        conversation_id,
        "Hello",
    )

    assert result.metadata["provider"] == "atlas"
    assert result.metadata["model"] == "atlas"
    assert result.metadata["backend"] == "development"


@pytest.mark.anyio
async def test_kernel_chat_preserves_backend_metadata() -> None:
    kernel = Kernel()
    kernel.boot()

    conversation_id, _ = kernel.create_conversation()

    result = await kernel.chat(
        conversation_id,
        "Hello",
    )

    assert result.metadata["mode"] == "development"


@pytest.mark.anyio
async def test_kernel_chat_forwards_generation_kwargs() -> None:
    kernel = Kernel()
    kernel.boot()

    conversation_id, _ = kernel.create_conversation()

    result = await kernel.chat(
        conversation_id,
        "Analyze BTC",
        temperature=0.2,
        max_tokens=300,
    )

    assert result.success is True


@pytest.mark.anyio
async def test_kernel_chat_raises_for_missing_conversation() -> None:
    kernel = Kernel()
    kernel.boot()

    with pytest.raises(
            KeyError,
            match="was not found",
    ):
        await kernel.chat(
            uuid4(),
            "Hello",
        )


def test_kernel_get_conversation_raises_for_missing_id() -> None:
    kernel = Kernel()
    kernel.boot()

    with pytest.raises(
            KeyError,
            match="was not found",
    ):
        kernel.get_conversation(uuid4())


@pytest.mark.anyio
async def test_kernel_tool_model_and_chat_runtimes_work_together() -> None:
    kernel = Kernel()
    kernel.boot()

    tool_result = await kernel.execute_tool(
        "system_info"
    )

    model_result = await kernel.generate(
        "echo",
        "Hello Echo",
    )

    conversation_id, _ = kernel.create_conversation()

    chat_result = await kernel.chat(
        conversation_id,
        "Hello Atlas",
    )

    assert tool_result.success is True
    assert model_result.success is True
    assert model_result.content == "Hello Echo"
    assert chat_result.success is True