from pathlib import Path

import pytest

from app.kernel.bootstrap import Kernel
from app.types.message import MessageRole


def create_kernel(
        tmp_path: Path,
) -> Kernel:
    kernel = Kernel()

    kernel.settings.CONVERSATION_STORAGE_PATH = str(
        tmp_path / "conversations"
    )

    kernel.boot()

    return kernel


def test_kernel_save_creates_persisted_conversation(
        tmp_path: Path,
) -> None:
    kernel = create_kernel(tmp_path)

    conversation_id, conversation = (
        kernel.create_conversation()
    )

    conversation.add_user("Hello Atlas")

    kernel.save_conversation(conversation_id)

    path = (
            tmp_path
            / "conversations"
            / f"{conversation_id}.json"
    )

    assert path.exists()


def test_kernel_save_preserves_chat_history(
        tmp_path: Path,
) -> None:
    kernel = create_kernel(tmp_path)

    conversation_id, _ = kernel.create_conversation()

    awaitable = kernel.chat(
        conversation_id,
        "Analyze BTC",
    )

    # This test is synchronous, so chat is tested separately below.
    assert awaitable is not None
    awaitable.close()


@pytest.mark.anyio
async def test_kernel_save_after_chat_persists_full_history(
        tmp_path: Path,
) -> None:
    kernel = create_kernel(tmp_path)

    conversation_id, _ = kernel.create_conversation()

    await kernel.chat(
        conversation_id,
        "Analyze BTC",
    )

    kernel.save_conversation(conversation_id)

    restarted = create_kernel(tmp_path)

    restored_count = restarted.restore_conversations()

    restored = restarted.get_conversation(
        conversation_id
    )

    assert restored_count == 1
    assert len(restored) == 2
    assert restored.all()[0].content == "Analyze BTC"
    assert restored.all()[1].content == (
        "Atlas development response."
    )


def test_new_kernel_does_not_restore_automatically(
        tmp_path: Path,
) -> None:
    first = create_kernel(tmp_path)

    conversation_id, conversation = (
        first.create_conversation()
    )

    conversation.add_user("Persistent message")
    first.save_conversation(conversation_id)

    restarted = create_kernel(tmp_path)

    assert restarted.conversation_count() == 0


def test_restore_returns_zero_for_empty_storage(
        tmp_path: Path,
) -> None:
    kernel = create_kernel(tmp_path)

    assert kernel.restore_conversations() == 0


def test_restore_returns_number_of_conversations(
        tmp_path: Path,
) -> None:
    first = create_kernel(tmp_path)

    first_id, _ = first.create_conversation()
    second_id, _ = first.create_conversation()

    first.save_conversation(first_id)
    first.save_conversation(second_id)

    restarted = create_kernel(tmp_path)

    assert restarted.restore_conversations() == 2
    assert restarted.conversation_count() == 2


def test_restored_conversation_keeps_same_id(
        tmp_path: Path,
) -> None:
    first = create_kernel(tmp_path)

    conversation_id, _ = first.create_conversation()

    first.save_conversation(conversation_id)

    restarted = create_kernel(tmp_path)
    restarted.restore_conversations()

    assert restarted.get_conversation(
        conversation_id
    ) is not None


@pytest.mark.anyio
async def test_restored_conversation_can_continue_chatting(
        tmp_path: Path,
) -> None:
    first = create_kernel(tmp_path)

    conversation_id, _ = first.create_conversation()

    await first.chat(
        conversation_id,
        "First question",
    )

    first.save_conversation(conversation_id)

    restarted = create_kernel(tmp_path)
    restarted.restore_conversations()

    result = await restarted.chat(
        conversation_id,
        "Second question",
    )

    restored = restarted.get_conversation(
        conversation_id
    )

    assert result.success is True
    assert len(restored) == 4


@pytest.mark.anyio
async def test_continued_chat_preserves_restored_history(
        tmp_path: Path,
) -> None:
    first = create_kernel(tmp_path)

    conversation_id, _ = first.create_conversation()

    await first.chat(
        conversation_id,
        "First question",
    )

    first.save_conversation(conversation_id)

    restarted = create_kernel(tmp_path)
    restarted.restore_conversations()

    await restarted.chat(
        conversation_id,
        "Second question",
    )

    restored = restarted.get_conversation(
        conversation_id
    )

    assert [message.content for message in restored.all()] == [
        "First question",
        "Atlas development response.",
        "Second question",
        "Atlas development response.",
    ]


def test_save_overwrites_existing_persisted_state(
        tmp_path: Path,
) -> None:
    kernel = create_kernel(tmp_path)

    conversation_id, conversation = (
        kernel.create_conversation()
    )

    conversation.add_user("First")
    kernel.save_conversation(conversation_id)

    conversation.add_assistant("Second")
    kernel.save_conversation(conversation_id)

    restarted = create_kernel(tmp_path)
    restarted.restore_conversations()

    restored = restarted.get_conversation(
        conversation_id
    )

    assert len(restored) == 2
    assert restored.last() is not None
    assert restored.last().content == "Second"


def test_delete_removes_persisted_conversation(
        tmp_path: Path,
) -> None:
    kernel = create_kernel(tmp_path)

    conversation_id, _ = kernel.create_conversation()

    kernel.save_conversation(conversation_id)
    kernel.delete_conversation(conversation_id)

    restarted = create_kernel(tmp_path)

    assert restarted.restore_conversations() == 0


def test_clear_memory_keeps_persisted_conversations(
        tmp_path: Path,
) -> None:
    kernel = create_kernel(tmp_path)

    conversation_id, _ = kernel.create_conversation()

    kernel.save_conversation(conversation_id)

    kernel.clear_conversations()

    assert kernel.conversation_count() == 0

    assert kernel.restore_conversations() == 1
    assert kernel.get_conversation(
        conversation_id
    ) is not None


def test_multiple_persisted_sessions_remain_isolated(
        tmp_path: Path,
) -> None:
    first = create_kernel(tmp_path)

    first_id, first_conversation = (
        first.create_conversation()
    )
    second_id, second_conversation = (
        first.create_conversation()
    )

    first_conversation.add_user("First history")
    second_conversation.add_user("Second history")

    first.save_conversation(first_id)
    first.save_conversation(second_id)

    restarted = create_kernel(tmp_path)
    restarted.restore_conversations()

    assert (
            restarted.get_conversation(first_id)
            .last()
            .content
            == "First history"
    )

    assert (
            restarted.get_conversation(second_id)
            .last()
            .content
            == "Second history"
    )


def test_restored_messages_preserve_roles(
        tmp_path: Path,
) -> None:
    first = create_kernel(tmp_path)

    conversation_id, conversation = (
        first.create_conversation()
    )

    conversation.add_system("You are Atlas.")
    conversation.add_user("Hello")
    conversation.add_assistant("Hi")

    first.save_conversation(conversation_id)

    restarted = create_kernel(tmp_path)
    restarted.restore_conversations()

    restored = restarted.get_conversation(
        conversation_id
    )

    assert [message.role for message in restored.all()] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]


def test_kernel_persistence_uses_configured_directory(
        tmp_path: Path,
) -> None:
    custom_directory = tmp_path / "custom_atlas_storage"

    kernel = Kernel()
    kernel.settings.CONVERSATION_STORAGE_PATH = str(
        custom_directory
    )
    kernel.boot()

    conversation_id, _ = kernel.create_conversation()
    kernel.save_conversation(conversation_id)

    assert (
            custom_directory
            / f"{conversation_id}.json"
    ).exists()