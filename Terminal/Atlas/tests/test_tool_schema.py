from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from app.tools.tool_schema import ToolSchema


def test_stores_name() -> None:
    schema = ToolSchema(
        name="system_info",
        description="Returns system information.",
    )

    assert schema.name == "system_info"


def test_stores_description() -> None:
    schema = ToolSchema(
        name="system_info",
        description="Returns system information.",
    )

    assert schema.description == "Returns system information."


def test_stores_arguments() -> None:
    schema = ToolSchema(
        name="price",
        description="Gets latest market price.",
        arguments={
            "symbol": "Trading pair",
        },
    )

    assert schema.arguments == {
        "symbol": "Trading pair",
    }


def test_equal_schemas() -> None:
    left = ToolSchema(
        "price",
        "Gets latest market price.",
        {
            "symbol": "Trading pair",
        },
    )

    right = ToolSchema(
        "price",
        "Gets latest market price.",
        {
            "symbol": "Trading pair",
        },
    )

    assert left == right


def test_different_names() -> None:
    left = ToolSchema(
        "price",
        "Gets latest market price.",
    )

    right = ToolSchema(
        "system_info",
        "Gets latest market price.",
    )

    assert left != right


def test_different_arguments() -> None:
    left = ToolSchema(
        "price",
        "Gets latest market price.",
        {
            "symbol": "Trading pair",
        },
    )

    right = ToolSchema(
        "price",
        "Gets latest market price.",
        {
            "market": "Market name",
        },
    )

    assert left != right


def test_frozen() -> None:
    schema = ToolSchema(
        "system_info",
        "Returns system information.",
    )

    with pytest.raises(FrozenInstanceError):
        schema.name = "cpu"


def test_hashable() -> None:
    schema = ToolSchema(
        "system_info",
        "Returns system information.",
    )

    assert hash(schema)


def test_repr() -> None:
    schema = ToolSchema(
        "system_info",
        "Returns system information.",
    )

    assert "ToolSchema" in repr(schema)


def test_empty_arguments() -> None:
    schema = ToolSchema(
        "system_info",
        "Returns system information.",
    )

    assert schema.arguments == {}


def test_multiple_arguments() -> None:
    schema = ToolSchema(
        "price",
        "Gets latest market price.",
        {
            "symbol": "Trading pair",
            "exchange": "Exchange name",
        },
    )

    assert schema.arguments == {
        "symbol": "Trading pair",
        "exchange": "Exchange name",
    }


def test_unicode_argument_names() -> None:
    schema = ToolSchema(
        "translate",
        "Translate text.",
        {
            "मजकूर": "Input text",
        },
    )

    assert schema.arguments == {
        "मजकूर": "Input text",
    }


def test_unicode_description() -> None:
    schema = ToolSchema(
        "translate",
        "मराठी आणि 日本語 translation 🚀",
    )

    assert schema.description == "मराठी आणि 日本語 translation 🚀"