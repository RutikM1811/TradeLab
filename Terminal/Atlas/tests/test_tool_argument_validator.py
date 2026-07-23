from __future__ import annotations

from app.tools.tool_argument_validator import ToolArgumentValidator
from app.tools.tool_call import ToolCall
from app.tools.tool_schema import ToolSchema


def test_valid_required_parameter() -> None:
    validator = ToolArgumentValidator()

    schema = ToolSchema(
        "price",
        "Returns price.",
        {
            "symbol": "Trading symbol",
        },
    )

    call = ToolCall(
        "price",
        {
            "symbol": "BTCUSDT",
        },
    )

    assert validator.validate(call, schema) == []


def test_missing_required_parameter() -> None:
    validator = ToolArgumentValidator()

    schema = ToolSchema(
        "price",
        "Returns price.",
        {
            "symbol": "Trading symbol",
        },
    )

    call = ToolCall(
        "price",
        {},
    )

    assert validator.validate(
        call,
        schema,
    ) == [
               "Missing required parameter: symbol",
           ]


def test_multiple_required_parameters() -> None:
    validator = ToolArgumentValidator()

    schema = ToolSchema(
        "price",
        "Returns price.",
        {
            "symbol": "",
            "exchange": "",
            "interval": "",
        },
    )

    call = ToolCall(
        "price",
        {},
    )

    assert validator.validate(
        call,
        schema,
    ) == [
               "Missing required parameter: symbol",
               "Missing required parameter: exchange",
               "Missing required parameter: interval",
           ]


def test_unknown_parameter_allowed() -> None:
    validator = ToolArgumentValidator()

    schema = ToolSchema(
        "price",
        "Returns price.",
        {
            "symbol": "",
        },
    )

    call = ToolCall(
        "price",
        {
            "symbol": "BTCUSDT",
            "foo": "bar",
        },
    )

    assert validator.validate(call, schema) == []


def test_no_parameters() -> None:
    validator = ToolArgumentValidator()

    schema = ToolSchema(
        "system_info",
        "Returns system information.",
        {},
    )

    call = ToolCall(
        "system_info",
        {},
    )

    assert validator.validate(call, schema) == []


def test_empty_schema() -> None:
    validator = ToolArgumentValidator()

    schema = ToolSchema(
        "",
        "",
        {},
    )

    call = ToolCall(
        "",
        {},
    )

    assert validator.validate(call, schema) == []


def test_empty_arguments() -> None:
    validator = ToolArgumentValidator()

    schema = ToolSchema(
        "tool",
        "description",
        {
            "a": "",
            "b": "",
        },
    )

    call = ToolCall(
        "tool",
        {},
    )

    assert len(
        validator.validate(
            call,
            schema,
        )
    ) == 2


def test_unicode_parameter_names() -> None:
    validator = ToolArgumentValidator()

    schema = ToolSchema(
        "unicode",
        "unicode",
        {
            "नाम": "",
        },
    )

    call = ToolCall(
        "unicode",
        {
            "नाम": "Atlas",
        },
    )

    assert validator.validate(call, schema) == []


def test_multiple_errors_returned() -> None:
    validator = ToolArgumentValidator()

    schema = ToolSchema(
        "tool",
        "description",
        {
            "a": "",
            "b": "",
            "c": "",
        },
    )

    call = ToolCall(
        "tool",
        {
            "a": "1",
        },
    )

    assert validator.validate(
        call,
        schema,
    ) == [
               "Missing required parameter: b",
               "Missing required parameter: c",
           ]


def test_all_parameters_present() -> None:
    validator = ToolArgumentValidator()

    schema = ToolSchema(
        "tool",
        "description",
        {
            "a": "",
            "b": "",
            "c": "",
        },
    )

    call = ToolCall(
        "tool",
        {
            "a": "1",
            "b": "2",
            "c": "3",
        },
    )

    assert validator.validate(call, schema) == []


def test_extra_parameters_do_not_fail() -> None:
    validator = ToolArgumentValidator()

    schema = ToolSchema(
        "tool",
        "description",
        {
            "symbol": "",
        },
    )

    call = ToolCall(
        "tool",
        {
            "symbol": "BTCUSDT",
            "exchange": "BINANCE",
            "limit": 100,
        },
    )

    assert validator.validate(call, schema) == []


def test_validator_returns_list() -> None:
    validator = ToolArgumentValidator()

    schema = ToolSchema(
        "tool",
        "description",
        {},
    )

    call = ToolCall(
        "tool",
        {},
    )

    result = validator.validate(
        call,
        schema,
    )

    assert isinstance(result, list)