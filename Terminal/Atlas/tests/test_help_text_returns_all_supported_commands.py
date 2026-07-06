from pathlib import Path
from uuid import uuid4

import pytest

from app.kernel.bootstrap import Kernel
from app.terminal.command_runtime import TerminalCommandRuntime


def create_runtime(
        tmp_path: Path,
) -> tuple[Kernel, TerminalCommandRuntime]:
    kernel = Kernel()

    kernel.settings.CONVERSATION_STORAGE_PATH = str(
        tmp_path / "conversations"
    )

    kernel.boot()

    runtime = TerminalCommandRuntime(kernel)

    return kernel, runtime


def test_create_conversation_creates_managed_session(
        tmp_path: Path,
) -> None:
    kernel, runtime = create_runtime(tmp_path)

    conversation_id, conversation = (
        runtime.create_conversation()
    )

    assert kernel.get_conversation(
        conversation_id
    ) is conversation


def test_create_conversation_increases_count(
        tmp_path: Path,
) -> None:
    kernel, runtime = create_runtime(tmp_path)

    runtime.create_conversation()

    assert kernel.conversation_count() == 1


def test_list_conversations_returns_all_sessions(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    first_id, first = runtime.create_conversation()
    second_id, second = runtime.create_conversation()

    assert runtime.list_conversations() == (
        (first_id, first),
        (second_id, second),
    )


def test_list_conversations_is_empty_initially(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    assert runtime.list_conversations() == ()


def test_switch_conversation_uses_one_based_index(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    first = runtime.create_conversation()
    second = runtime.create_conversation()

    assert runtime.switch_conversation(1) == first
    assert runtime.switch_conversation(2) == second


def test_switch_conversation_preserves_creation_order(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    first = runtime.create_conversation()
    runtime.create_conversation()
    third = runtime.create_conversation()

    assert runtime.switch_conversation(1) == first
    assert runtime.switch_conversation(3) == third


def test_switch_rejects_zero_index(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    runtime.create_conversation()

    with pytest.raises(
            IndexError,
            match="Conversation number 0 does not exist",
    ):
        runtime.switch_conversation(0)


def test_switch_rejects_negative_index(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    runtime.create_conversation()

    with pytest.raises(IndexError):
        runtime.switch_conversation(-1)


def test_switch_rejects_index_above_count(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    runtime.create_conversation()

    with pytest.raises(
            IndexError,
            match="Conversation number 2 does not exist",
    ):
        runtime.switch_conversation(2)


def test_history_returns_empty_tuple_for_empty_conversation(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    conversation_id, _ = runtime.create_conversation()

    assert runtime.history(conversation_id) == ()


def test_history_formats_all_message_roles(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    conversation_id, conversation = (
        runtime.create_conversation()
    )

    conversation.add_system("You are Atlas.")
    conversation.add_user("Analyze BTC")
    conversation.add_tool("BTC price is 65000")
    conversation.add_assistant("BTC is bullish.")

    assert runtime.history(conversation_id) == (
        "System: You are Atlas.",
        "User: Analyze BTC",
        "Tool: BTC price is 65000",
        "Assistant: BTC is bullish.",
    )


def test_history_preserves_message_order(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    conversation_id, conversation = (
        runtime.create_conversation()
    )

    conversation.add_user("First")
    conversation.add_assistant("Second")
    conversation.add_user("Third")

    assert runtime.history(conversation_id) == (
        "User: First",
        "Assistant: Second",
        "User: Third",
    )


def test_delete_conversation_removes_session(
        tmp_path: Path,
) -> None:
    kernel, runtime = create_runtime(tmp_path)

    conversation_id, _ = runtime.create_conversation()

    runtime.delete_conversation(conversation_id)

    assert kernel.conversation_count() == 0


def test_delete_conversation_removes_persisted_session(
        tmp_path: Path,
) -> None:
    kernel, runtime = create_runtime(tmp_path)

    conversation_id, _ = runtime.create_conversation()

    kernel.save_conversation(conversation_id)

    runtime.delete_conversation(conversation_id)

    restarted, _ = create_runtime(tmp_path)

    assert restarted.restore_conversations() == 0


def test_help_text_returns_all_supported_commands(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    help_text = runtime.help_text()

    assert help_text == (
        "/new - Create a new conversation",
        "/list - Show all conversations",
        "/switch <number> - Switch conversation",
        "/history - Show current conversation history",
        "/info - Show current conversation information",
        "/rename <title> - Rename current conversation",
        "/delete - Delete current conversation",
        "/help - Show available commands",
        "/exit - Save and stop Atlas",
    )


def test_rename_conversation_changes_title(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    conversation_id, conversation = (
        runtime.create_conversation()
    )

    runtime.rename_conversation(
        conversation_id,
        "BTC Research",
    )

    assert conversation.metadata.title == "BTC Research"


def test_rename_conversation_rejects_empty_title(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    conversation_id, _ = runtime.create_conversation()

    with pytest.raises(
            ValueError,
            match="Conversation title cannot be empty",
    ):
        runtime.rename_conversation(
            conversation_id,
            "",
        )


def test_rename_conversation_rejects_whitespace_title(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    conversation_id, _ = runtime.create_conversation()

    with pytest.raises(ValueError):
        runtime.rename_conversation(
            conversation_id,
            "   ",
        )


def test_rename_conversation_preserves_messages(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    conversation_id, conversation = (
        runtime.create_conversation()
    )

    conversation.add_user("Analyze BTC")

    runtime.rename_conversation(
        conversation_id,
        "Bitcoin Research",
    )

    assert len(conversation) == 1
    assert conversation.last() is not None
    assert conversation.last().content == "Analyze BTC"


def test_rename_conversation_is_persisted(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    conversation_id, _ = runtime.create_conversation()

    runtime.rename_conversation(
        conversation_id,
        "Persistent Title",
    )

    restarted_kernel, _ = create_runtime(tmp_path)

    restarted_kernel.restore_conversations()

    restored = restarted_kernel.get_conversation(
        conversation_id
    )

    assert restored.metadata.title == "Persistent Title"


def test_rename_can_replace_automatic_title(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    conversation_id, conversation = (
        runtime.create_conversation()
    )

    conversation.metadata.rename("Automatic Title")

    runtime.rename_conversation(
        conversation_id,
        "Custom Title",
    )

    assert conversation.metadata.title == "Custom Title"


def test_rename_missing_conversation_raises(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    with pytest.raises(
            KeyError,
            match="was not found",
    ):
        runtime.rename_conversation(
            uuid4(),
            "Missing Chat",
        )


def test_conversation_info_returns_id(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    conversation_id, _ = runtime.create_conversation()

    info = runtime.conversation_info(
        conversation_id
    )

    assert info["id"] == conversation_id


def test_conversation_info_returns_title(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    conversation_id, conversation = (
        runtime.create_conversation()
    )

    conversation.metadata.rename("BTC Analysis")

    info = runtime.conversation_info(
        conversation_id
    )

    assert info["title"] == "BTC Analysis"


def test_conversation_info_returns_timestamps(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    conversation_id, conversation = (
        runtime.create_conversation()
    )

    info = runtime.conversation_info(
        conversation_id
    )

    assert (
            info["created_at"]
            == conversation.metadata.created_at
    )
    assert (
            info["updated_at"]
            == conversation.metadata.updated_at
    )


def test_conversation_info_returns_zero_message_count(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    conversation_id, _ = runtime.create_conversation()

    info = runtime.conversation_info(
        conversation_id
    )

    assert info["message_count"] == 0


def test_conversation_info_returns_current_message_count(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    conversation_id, conversation = (
        runtime.create_conversation()
    )

    conversation.add_user("Hello")
    conversation.add_assistant("Hi")

    info = runtime.conversation_info(
        conversation_id
    )

    assert info["message_count"] == 2


def test_conversation_info_reflects_renamed_title(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    conversation_id, _ = runtime.create_conversation()

    runtime.rename_conversation(
        conversation_id,
        "Market Research",
    )

    info = runtime.conversation_info(
        conversation_id
    )

    assert info["title"] == "Market Research"


def test_conversation_info_raises_for_missing_conversation(
        tmp_path: Path,
) -> None:
    _, runtime = create_runtime(tmp_path)

    with pytest.raises(
            KeyError,
            match="was not found",
    ):
        runtime.conversation_info(
            uuid4()
        )