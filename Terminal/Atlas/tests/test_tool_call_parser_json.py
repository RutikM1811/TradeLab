from __future__ import annotations

import json

from app.tools.tool_call import ToolCall
from app.tools.tool_call_parser import ToolCallParser


def test_parse_json_tool_call() -> None:
    parser = ToolCallParser()

    result = parser.parse(
        json.dumps(
            {
                "tool": "system_info",
                "arguments": {},
            }
        )
    )

    assert result == ToolCall(
        tool_name="system_info",
        arguments={},
    )


def test_parse_json_with_arguments() -> None:
    parser = ToolCallParser()

    result = parser.parse(
        json.dumps(
            {
                "tool": "price",
                "arguments": {
                    "symbol": "BTCUSDT",
                },
            }
        )
    )

    assert result == ToolCall(
        tool_name="price",
        arguments={
            "symbol": "BTCUSDT",
        },
    )


def test_parse_json_with_multiple_arguments() -> None:
    parser = ToolCallParser()

    result = parser.parse(
        json.dumps(
            {
                "tool": "price",
                "arguments": {
                    "symbol": "BTCUSDT",
                    "exchange": "BINANCE",
                    "limit": 10,
                },
            }
        )
    )

    assert result == ToolCall(
        tool_name="price",
        arguments={
            "symbol": "BTCUSDT",
            "exchange": "BINANCE",
            "limit": 10,
        },
    )


def test_parse_json_empty_arguments() -> None:
    parser = ToolCallParser()

    result = parser.parse(
        json.dumps(
            {
                "tool": "system_info",
                "arguments": {},
            }
        )
    )

    assert result.arguments == {}


def test_invalid_json_returns_none() -> None:
    parser = ToolCallParser()

    result = parser.parse(
        '{"tool":"system_info","arguments":'
    )

    assert result is None


def test_missing_tool_returns_none() -> None:
    parser = ToolCallParser()

    result = parser.parse(
        json.dumps(
            {
                "arguments": {},
            }
        )
    )

    assert result is None


def test_missing_arguments_defaults_to_empty_dict() -> None:
    parser = ToolCallParser()

    result = parser.parse(
        json.dumps(
            {
                "tool": "system_info",
            }
        )
    )

    assert result == ToolCall(
        tool_name="system_info",
        arguments={},
    )


def test_tool_must_be_string() -> None:
    parser = ToolCallParser()

    result = parser.parse(
        json.dumps(
            {
                "tool": 123,
                "arguments": {},
            }
        )
    )

    assert result is None


def test_arguments_must_be_object() -> None:
    parser = ToolCallParser()

    result = parser.parse(
        json.dumps(
            {
                "tool": "system_info",
                "arguments": [],
            }
        )
    )

    assert result is None


def test_empty_json_returns_none() -> None:
    parser = ToolCallParser()

    result = parser.parse("{}")

    assert result is None