from uuid import uuid4

import pytest

from app.kernel.bootstrap import Kernel


def test_new_kernel_has_no_conversations() -> None:
    kernel = Kernel()
    kernel.boot()

    assert kernel.conversation_count() == 0
    assert kernel.list_conversations() == ()


def test_create_increases_conversation_count() -> None:
    kernel = Kernel()
    kernel.boot()

    kernel.create_conversation()

    assert kernel.conversation_count() == 1


def test_multiple_creations_increase_count() -> None:
    kernel = Kernel()
    kernel.boot()

    kernel.create_conversation()
    kernel.create_conversation()
    kernel.create_conversation()

    assert kernel.conversation_count() == 3


def test_list_returns_created_conversations() -> None:
    kernel = Kernel()
    kernel.boot()

    first_id, first = kernel.create_conversation()
    second_id, second = kernel.create_conversation()

    assert kernel.list_conversations() == (
        (first_id, first),
        (second_id, second),
    )


def test_list_preserves_creation_order() -> None:
    kernel = Kernel()
    kernel.boot()

    first_id, _ = kernel.create_conversation()
    second_id, _ = kernel.create_conversation()
    third_id, _ = kernel.create_conversation()

    ids = tuple(
        conversation_id
        for conversation_id, _ in kernel.list_conversations()
    )

    assert ids == (
        first_id,
        second_id,
        third_id,
    )


def test_list_returns_tuple() -> None:
    kernel = Kernel()
    kernel.boot()

    kernel.create_conversation()

    assert isinstance(kernel.list_conversations(), tuple)


def test_delete_removes_conversation() -> None:
    kernel = Kernel()
    kernel.boot()

    conversation_id, _ = kernel.create_conversation()

    kernel.delete_conversation(conversation_id)

    assert kernel.conversation_count() == 0
    assert kernel.list_conversations() == ()


def test_delete_only_removes_target_conversation() -> None:
    kernel = Kernel()
    kernel.boot()

    first_id, _ = kernel.create_conversation()
    second_id, second = kernel.create_conversation()

    kernel.delete_conversation(first_id)

    assert kernel.conversation_count() == 1
    assert kernel.get_conversation(second_id) is second


def test_deleted_conversation_cannot_be_resolved() -> None:
    kernel = Kernel()
    kernel.boot()

    conversation_id, _ = kernel.create_conversation()

    kernel.delete_conversation(conversation_id)

    with pytest.raises(
            KeyError,
            match="was not found",
    ):
        kernel.get_conversation(conversation_id)


@pytest.mark.anyio
async def test_deleted_conversation_cannot_receive_chat() -> None:
    kernel = Kernel()
    kernel.boot()

    conversation_id, _ = kernel.create_conversation()

    kernel.delete_conversation(conversation_id)

    with pytest.raises(
            KeyError,
            match="was not found",
    ):
        await kernel.chat(
            conversation_id,
            "Hello",
        )


def test_delete_raises_for_missing_conversation() -> None:
    kernel = Kernel()
    kernel.boot()

    with pytest.raises(
            KeyError,
            match="was not found",
    ):
        kernel.delete_conversation(uuid4())


def test_clear_removes_all_conversations() -> None:
    kernel = Kernel()
    kernel.boot()

    kernel.create_conversation()
    kernel.create_conversation()
    kernel.create_conversation()

    kernel.clear_conversations()

    assert kernel.conversation_count() == 0
    assert kernel.list_conversations() == ()


def test_clear_on_empty_kernel_is_safe() -> None:
    kernel = Kernel()
    kernel.boot()

    kernel.clear_conversations()

    assert kernel.conversation_count() == 0


def test_create_works_after_clear() -> None:
    kernel = Kernel()
    kernel.boot()

    kernel.create_conversation()
    kernel.clear_conversations()

    conversation_id, conversation = (
        kernel.create_conversation()
    )

    assert kernel.conversation_count() == 1
    assert kernel.get_conversation(
        conversation_id
    ) is conversation


@pytest.mark.anyio
async def test_chat_history_survives_listing() -> None:
    kernel = Kernel()
    kernel.boot()

    conversation_id, conversation = (
        kernel.create_conversation()
    )

    await kernel.chat(
        conversation_id,
        "Hello Atlas",
    )

    listed_id, listed_conversation = (
        kernel.list_conversations()[0]
    )

    assert listed_id == conversation_id
    assert listed_conversation is conversation
    assert len(listed_conversation) == 2


@pytest.mark.anyio
async def test_deleting_one_chat_preserves_other_history() -> None:
    kernel = Kernel()
    kernel.boot()

    first_id, _ = kernel.create_conversation()
    second_id, second = kernel.create_conversation()

    await kernel.chat(
        first_id,
        "First conversation",
    )

    await kernel.chat(
        second_id,
        "Second conversation",
    )

    kernel.delete_conversation(first_id)

    assert kernel.conversation_count() == 1
    assert len(second) == 2
    assert second.all()[0].content == "Second conversation"


def test_conversation_count_matches_list_length() -> None:
    kernel = Kernel()
    kernel.boot()

    kernel.create_conversation()
    kernel.create_conversation()

    assert kernel.conversation_count() == len(
        kernel.list_conversations()
    )


@pytest.mark.anyio
async def test_new_chat_works_after_clear() -> None:
    kernel = Kernel()
    kernel.boot()

    old_id, _ = kernel.create_conversation()

    await kernel.chat(
        old_id,
        "Old conversation",
    )

    kernel.clear_conversations()

    new_id, new_conversation = (
        kernel.create_conversation()
    )

    result = await kernel.chat(
        new_id,
        "New conversation",
    )

    assert result.success is True
    assert kernel.conversation_count() == 1
    assert len(new_conversation) == 2
    assert new_conversation.all()[0].content == (
        "New conversation"
    )