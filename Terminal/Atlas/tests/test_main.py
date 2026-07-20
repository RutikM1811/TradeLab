import builtins

import pytest

from app.main import main
from app.types.model_result import ModelResult


class DummyMetadata:
    def __init__(self, title):
        self.title = title


class DummyConversation:
    def __init__(self, title, message_count):
        self.metadata = DummyMetadata(title)
        self._message_count = message_count

    def __len__(self):
        return self._message_count


class DummyKernel:
    def __init__(self):
        self.boot_called = False
        self.restore_count = 0
        self.saved = []
        self.chat_calls = []
        self.chat_result = ModelResult(
            success=True,
            content="hello",
            error=None,
        )

    def boot(self):
        self.boot_called = True

    def restore_conversations(self):
        return self.restore_count

    def list_conversations(self):
        return [("conv-1", object())]

    async def chat(self, conversation_id, message):
        self.chat_calls.append((conversation_id, message))
        return self.chat_result

    def save_conversation(self, conversation_id):
        self.saved.append(conversation_id)


class DummyCommands:
    def __init__(self, kernel):
        self.kernel = kernel
        self.created = 0

    def create_conversation(self):
        self.created += 1
        return ("new-conv", object())

    def help_text(self):
        return ["help"]

    def list_conversations(self):
        return []

    def history(self, _):
        return []

    def delete_conversation(self, _):
        pass

    def rename_conversation(self, *_):
        pass

    def switch_conversation(self, *_):
        return ("new-conv", object())

    def conversation_info(self, _):
        return {
            "title": "Test",
            "id": "new-conv",
            "created_at": "today",
            "updated_at": "today",
            "message_count": 0,
        }


@pytest.fixture
def kernel(monkeypatch):
    instance = DummyKernel()

    monkeypatch.setattr(
        "app.main.Kernel",
        lambda: instance,
    )

    return instance


@pytest.fixture
def commands(kernel, monkeypatch):
    instance = DummyCommands(kernel)

    monkeypatch.setattr(
        "app.main.TerminalCommandRuntime",
        lambda _: instance,
    )

    return {"instance": instance}


def set_inputs(monkeypatch, values):
    """Feed a sequence of input() return values to main()."""

    iterator = iter(values)

    monkeypatch.setattr(
        builtins,
        "input",
        lambda _: next(iterator),
    )


@pytest.mark.anyio
async def test_boot_called(kernel, commands, monkeypatch):
    set_inputs(monkeypatch, ["/exit"])

    await main()

    assert kernel.boot_called


@pytest.mark.anyio
async def test_restores_existing_conversation(kernel, commands, monkeypatch):
    kernel.restore_count = 2

    set_inputs(monkeypatch, ["/exit"])

    await main()

    assert commands["instance"].created == 0


@pytest.mark.anyio
async def test_creates_new_conversation_when_none_restored(
        kernel,
        commands,
        monkeypatch,
):
    kernel.restore_count = 0

    set_inputs(monkeypatch, ["/exit"])

    await main()

    assert commands["instance"].created == 1


@pytest.mark.anyio
async def test_exit_command_slash(kernel, commands, monkeypatch):
    set_inputs(monkeypatch, ["/exit"])

    await main()

    assert kernel.saved == ["new-conv"]


@pytest.mark.anyio
async def test_exit_command_exit(kernel, commands, monkeypatch):
    set_inputs(monkeypatch, ["exit"])

    await main()

    assert kernel.saved == ["new-conv"]


@pytest.mark.anyio
async def test_exit_command_quit(kernel, commands, monkeypatch):
    set_inputs(monkeypatch, ["quit"])

    await main()

    assert kernel.saved == ["new-conv"]


@pytest.mark.anyio
async def test_empty_message_is_ignored(kernel, commands, monkeypatch):
    set_inputs(monkeypatch, ["", "/exit"])

    await main()

    assert kernel.chat_calls == []


@pytest.mark.anyio
async def test_keyboard_interrupt_exits(kernel, commands, monkeypatch):
    def raise_interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        builtins,
        "input",
        raise_interrupt,
    )

    await main()

    assert kernel.saved == ["new-conv"]


@pytest.mark.anyio
async def test_eof_error_exits(kernel, commands, monkeypatch):
    def raise_eof(_):
        raise EOFError

    monkeypatch.setattr(
        builtins,
        "input",
        raise_eof,
    )

    await main()

    assert kernel.saved == ["new-conv"]


@pytest.mark.anyio
async def test_startup_banner_printed(
        kernel,
        commands,
        monkeypatch,
        capsys,
):
    set_inputs(monkeypatch, ["/exit"])

    await main()

    output = capsys.readouterr().out

    assert "Atlas is ready." in output
    assert "Type /help" in output


@pytest.mark.anyio
async def test_shutdown_banner_printed(
        kernel,
        commands,
        monkeypatch,
        capsys,
):
    set_inputs(monkeypatch, ["/exit"])

    await main()

    output = capsys.readouterr().out

    assert "Conversation saved." in output
    assert "Atlas stopped." in output


# ---------------------------------------------------------------------------
# /help
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_help_command_prints_help_text(
        kernel,
        commands,
        monkeypatch,
        capsys,
):
    set_inputs(monkeypatch, ["/help", "/exit"])

    await main()

    output = capsys.readouterr().out

    assert "help" in output


# ---------------------------------------------------------------------------
# /new
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_new_command_creates_conversation(
        kernel,
        commands,
        monkeypatch,
        capsys,
):
    set_inputs(monkeypatch, ["/new", "/exit"])

    await main()

    output = capsys.readouterr().out

    # Once on startup (no restore), once for /new
    assert commands["instance"].created == 2
    assert "Created new conversation:" in output


# ---------------------------------------------------------------------------
# /list
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_list_command_with_no_conversations(
        kernel,
        commands,
        monkeypatch,
        capsys,
):
    set_inputs(monkeypatch, ["/list", "/exit"])

    await main()

    output = capsys.readouterr().out

    assert "No conversations." in output


@pytest.mark.anyio
async def test_list_command_with_conversations(
        kernel,
        commands,
        monkeypatch,
        capsys,
):
    set_inputs(monkeypatch, ["/list", "/exit"])

    conv1 = DummyConversation("First chat", 3)
    conv2 = DummyConversation("Second chat", 5)

    commands["instance"].list_conversations = lambda: [
        ("new-conv", conv1),
        ("other-conv", conv2),
    ]

    await main()

    output = capsys.readouterr().out

    assert "First chat" in output
    assert "Second chat" in output
    assert "* 1." in output
    assert "  2." in output


# ---------------------------------------------------------------------------
# /rename
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_rename_command_without_title_shows_usage(
        kernel,
        commands,
        monkeypatch,
        capsys,
):
    set_inputs(monkeypatch, ["/rename", "/exit"])

    await main()

    output = capsys.readouterr().out

    assert "Usage: /rename <title>" in output


@pytest.mark.anyio
async def test_rename_command_success(
        kernel,
        commands,
        monkeypatch,
        capsys,
):
    set_inputs(monkeypatch, ["/rename New Title", "/exit"])

    await main()

    output = capsys.readouterr().out

    assert "Conversation renamed to: New Title" in output


@pytest.mark.anyio
async def test_rename_command_handles_value_error(
        kernel,
        commands,
        monkeypatch,
        capsys,
):
    set_inputs(monkeypatch, ["/rename Bad Title", "/exit"])

    def raise_value_error(*_):
        raise ValueError("Title is invalid.")

    commands["instance"].rename_conversation = raise_value_error

    await main()

    output = capsys.readouterr().out

    assert "Command error: Title is invalid." in output


# ---------------------------------------------------------------------------
# /switch
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_switch_command_success(
        kernel,
        commands,
        monkeypatch,
        capsys,
):
    set_inputs(monkeypatch, ["/switch 1", "/exit"])

    await main()

    output = capsys.readouterr().out

    assert "Switched to conversation 1:" in output


@pytest.mark.anyio
async def test_switch_command_non_integer_index(
        kernel,
        commands,
        monkeypatch,
        capsys,
):
    set_inputs(monkeypatch, ["/switch abc", "/exit"])

    await main()

    output = capsys.readouterr().out

    assert "Command error:" in output


@pytest.mark.anyio
async def test_switch_command_out_of_range_index(
        kernel,
        commands,
        monkeypatch,
        capsys,
):
    set_inputs(monkeypatch, ["/switch 999", "/exit"])

    def raise_index_error(index):
        raise IndexError("No such conversation.")

    commands["instance"].switch_conversation = raise_index_error

    await main()

    output = capsys.readouterr().out

    assert "Command error: No such conversation." in output


# ---------------------------------------------------------------------------
# /history
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_history_command_when_empty(
        kernel,
        commands,
        monkeypatch,
        capsys,
):
    set_inputs(monkeypatch, ["/history", "/exit"])

    await main()

    output = capsys.readouterr().out

    assert "Conversation is empty." in output


@pytest.mark.anyio
async def test_history_command_with_messages(
        kernel,
        commands,
        monkeypatch,
        capsys,
):
    set_inputs(monkeypatch, ["/history", "/exit"])

    commands["instance"].history = lambda _: [
        "User: Hi",
        "Atlas: Hello",
    ]

    await main()

    output = capsys.readouterr().out

    assert "User: Hi" in output
    assert "Atlas: Hello" in output


# ---------------------------------------------------------------------------
# /delete
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_delete_command_creates_new_when_none_remain(
        kernel,
        commands,
        monkeypatch,
        capsys,
):
    set_inputs(monkeypatch, ["/delete", "/exit"])

    # Default DummyCommands.list_conversations() already returns [].
    await main()

    output = capsys.readouterr().out

    assert "Current conversation deleted." in output
    assert "Created new conversation:" in output


@pytest.mark.anyio
async def test_delete_command_switches_to_remaining_conversation(
        kernel,
        commands,
        monkeypatch,
        capsys,
):
    set_inputs(monkeypatch, ["/delete", "/exit"])

    remaining = DummyConversation("Remaining chat", 1)

    commands["instance"].list_conversations = lambda: [
        ("conv2", remaining),
    ]

    await main()

    output = capsys.readouterr().out

    assert "Current conversation deleted." in output
    assert "Switched to conversation: conv2" in output


# ---------------------------------------------------------------------------
# /info
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_info_command_prints_conversation_details(
        kernel,
        commands,
        monkeypatch,
        capsys,
):
    set_inputs(monkeypatch, ["/info", "/exit"])

    await main()

    output = capsys.readouterr().out

    assert "Title: Test" in output
    assert "ID: new-conv" in output
    assert "Created: today" in output
    assert "Updated: today" in output
    assert "Messages: 0" in output


# ---------------------------------------------------------------------------
# Unknown slash command
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_unknown_command_shows_error(
        kernel,
        commands,
        monkeypatch,
        capsys,
):
    set_inputs(monkeypatch, ["/bogus", "/exit"])

    await main()

    output = capsys.readouterr().out

    assert "Unknown command." in output


# ---------------------------------------------------------------------------
# Chat messages
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_chat_message_calls_kernel_and_saves(
        kernel,
        commands,
        monkeypatch,
        capsys,
):
    set_inputs(monkeypatch, ["Hello", "/exit"])

    await main()

    output = capsys.readouterr().out

    assert kernel.chat_calls == [("new-conv", "Hello")]
    assert "Atlas: hello" in output
    # Saved once after chat, once at shutdown.
    assert kernel.saved == ["new-conv", "new-conv"]


@pytest.mark.anyio
async def test_chat_message_prints_error_on_failure(
        kernel,
        commands,
        monkeypatch,
        capsys,
):
    set_inputs(monkeypatch, ["Hello", "/exit"])

    kernel.chat_result = ModelResult(
        success=False,
        content=None,
        error="Model unavailable.",
    )

    await main()

    output = capsys.readouterr().out

    assert "Atlas error: Model unavailable." in output


@pytest.mark.anyio
async def test_chat_message_prints_default_error_when_no_error_given(
        kernel,
        commands,
        monkeypatch,
        capsys,
):
    set_inputs(monkeypatch, ["Hello", "/exit"])

    kernel.chat_result = ModelResult(
        success=False,
        content=None,
        error=None,
    )

    await main()

    output = capsys.readouterr().out

    assert "Atlas error: No response generated." in output