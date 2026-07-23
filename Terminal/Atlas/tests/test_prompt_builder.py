from __future__ import annotations

from app.services.prompt_builder import PromptBuilder
from app.tools.tool_schema import ToolSchema


def test_build_without_tools() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        "Hello",
        (),
    )

    assert "Available tools:" not in prompt


def test_includes_user_message() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        "Hello Atlas",
        (),
    )

    assert "Hello Atlas" in prompt


def test_empty_message_supported() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        "",
        (),
    )

    assert "User message:" in prompt


def test_unicode_message() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        "नमस्कार 世界 🚀",
        (),
    )

    assert "नमस्कार 世界 🚀" in prompt


def test_single_tool_name() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        "Hi",
        (
            ToolSchema(
                "system_info",
                "Returns system information.",
                {},
            ),
        ),
    )

    assert "system_info" in prompt


def test_single_tool_description() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        "Hi",
        (
            ToolSchema(
                "system_info",
                "Returns system information.",
                {},
            ),
        ),
    )

    assert "Returns system information." in prompt


def test_single_tool_returns() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        "Hi",
        (
            ToolSchema(
                "system_info",
                "Returns system information.",
                {},
            ),
        ),
    )

    assert "Available tools:" in prompt


def test_single_tool_parameters() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        "Hi",
        (
            ToolSchema(
                "price",
                "Gets latest price.",
                {
                    "symbol": "Trading pair",
                },
            ),
        ),
    )

    assert "symbol" in prompt
    assert "Trading pair" in prompt


def test_multiple_tools() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        "Hi",
        (
            ToolSchema(
                "system_info",
                "Returns system information.",
                {},
            ),
            ToolSchema(
                "price",
                "Gets latest price.",
                {},
            ),
        ),
    )

    assert "system_info" in prompt
    assert "price" in prompt


def test_preserves_tool_order() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        "Hi",
        (
            ToolSchema(
                "first",
                "First tool.",
                {},
            ),
            ToolSchema(
                "second",
                "Second tool.",
                {},
            ),
        ),
    )

    assert prompt.index("first") < prompt.index("second")


def test_separates_tools() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        "Hi",
        (
            ToolSchema(
                "one",
                "Tool one.",
                {},
            ),
            ToolSchema(
                "two",
                "Tool two.",
                {},
            ),
        ),
    )

    assert "\n\n2." in prompt


def test_required_parameter() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        "Hi",
        (
            ToolSchema(
                "price",
                "Gets latest price.",
                {
                    "symbol": "Required",
                },
            ),
        ),
    )

    assert "symbol" in prompt


def test_optional_parameter() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        "Hi",
        (
            ToolSchema(
                "weather",
                "Gets weather.",
                {
                    "city": "Optional",
                },
            ),
        ),
    )

    assert "city" in prompt


def test_no_parameters() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        "Hi",
        (
            ToolSchema(
                "system_info",
                "Returns system information.",
                {},
            ),
        ),
    )

    assert "Arguments: None" in prompt


def test_deterministic_output() -> None:
    builder = PromptBuilder()

    tools = (
        ToolSchema(
            "system_info",
            "Returns system information.",
            {},
        ),
    )

    first = builder.build(
        "Hello",
        tools,
    )

    second = builder.build(
        "Hello",
        tools,
    )

    assert first == second


def test_no_extra_whitespace() -> None:
    builder = PromptBuilder()

    prompt = builder.build(
        "Hello",
        (),
    )

    assert prompt == prompt.strip()