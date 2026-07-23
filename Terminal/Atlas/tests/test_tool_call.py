from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from app.tools.tool_call import ToolCall


def test_tool_call_stores_tool_name() -> None:
    call = ToolCall(
        "system_info",
        {},
    )

    assert call.tool_name == "system_info"


def test_tool_call_stores_arguments() -> None:
    call = ToolCall(
        "price",
        {
            "symbol": "BTCUSDT",
        },
    )

    assert call.arguments == {
        "symbol": "BTCUSDT",
    }


def test_empty_arguments_supported() -> None:
    call = ToolCall(
        "system_info",
        {},
    )

    assert call.arguments == {}


def test_multiple_arguments_supported() -> None:
    call = ToolCall(
        "price",
        {
            "symbol": "BTCUSDT",
            "exchange": "BINANCE",
            "limit": 10,
        },
    )

    assert call.arguments == {
        "symbol": "BTCUSDT",
        "exchange": "BINANCE",
        "limit": 10,
    }


def test_tool_call_equality() -> None:
    left = ToolCall(
        "price",
        {
            "symbol": "BTCUSDT",
        },
    )

    right = ToolCall(
        "price",
        {
            "symbol": "BTCUSDT",
        },
    )

    assert left == right


def test_tool_call_is_hashable() -> None:
    call = ToolCall(
        "system_info",
        {},
    )

    assert hash(call)


def test_tool_call_is_frozen() -> None:
    call = ToolCall(
        "system_info",
        {},
    )

    with pytest.raises(FrozenInstanceError):
        call.tool_name = "cpu"


def test_repr_contains_tool_name() -> None:
    call = ToolCall(
        "system_info",
        {},
    )

    assert "system_info" in repr(call)


def test_repr_contains_arguments() -> None:
    call = ToolCall(
        "price",
        {
            "symbol": "BTCUSDT",
        },
    )

    representation = repr(call)

    assert "symbol" in representation
    assert "BTCUSDT" in representation


def test_supports_unicode_tool_name() -> None:
    call = ToolCall(
        "सिस्टम_माहिती",
        {},
    )

    assert call.tool_name == "सिस्टम_माहिती"


def test_argument_values_can_be_mixed_types() -> None:
    call = ToolCall(
        "example",
        {
            "text": "hello",
            "count": 5,
            "enabled": True,
            "value": 3.14,
        },
    )

    assert call.arguments["text"] == "hello"
    assert call.arguments["count"] == 5
    assert call.arguments["enabled"] is True
    assert call.arguments["value"] == 3.14


def test_empty_tool_name_is_allowed() -> None:
    call = ToolCall(
        "",
        {},
    )

    assert call.tool_name == ""