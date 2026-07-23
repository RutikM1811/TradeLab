from dataclasses import FrozenInstanceError

import pytest

from app.tools.tool_parameter import ToolParameter
from app.tools.tool_schema import ToolSchema


def test_schema_stores_name() -> None:
    schema = ToolSchema(
        name="system_info",
        description="Returns system information",
    )

    assert schema.name == "system_info"


def test_schema_stores_description() -> None:
    schema = ToolSchema(
        name="system_info",
        description="Returns system information",
    )

    assert schema.description == "Returns system information"


def test_parameters_default_to_empty_tuple() -> None:
    schema = ToolSchema(
        name="system_info",
        description="Returns system information",
    )

    assert schema.parameters == ()


def test_returns_defaults_to_empty_string() -> None:
    schema = ToolSchema(
        name="system_info",
        description="Returns system information",
    )

    assert schema.returns == ""


def test_schema_accepts_parameters() -> None:
    parameter = ToolParameter(
        name="symbol",
        description="Trading pair",
    )

    schema = ToolSchema(
        name="price",
        description="Returns latest price",
        parameters=(parameter,),
    )

    assert schema.parameters == (parameter,)


def test_schema_accepts_multiple_parameters() -> None:
    first = ToolParameter(
        "symbol",
        "Trading pair",
    )

    second = ToolParameter(
        "exchange",
        "Exchange",
        required=False,
    )

    schema = ToolSchema(
        name="price",
        description="Returns latest price",
        parameters=(first, second),
    )

    assert len(schema.parameters) == 2


def test_schema_accepts_returns_description() -> None:
    schema = ToolSchema(
        name="price",
        description="Returns latest price",
        returns="Current market price",
    )

    assert schema.returns == "Current market price"


def test_schema_is_frozen() -> None:
    schema = ToolSchema(
        name="system_info",
        description="Returns system information",
    )

    with pytest.raises(FrozenInstanceError):
        schema.name = "cpu"


def test_schema_is_hashable() -> None:
    schema = ToolSchema(
        name="system_info",
        description="Returns system information",
    )

    assert hash(schema)


def test_schema_equality() -> None:
    left = ToolSchema(
        "system_info",
        "Returns system information",
    )

    right = ToolSchema(
        "system_info",
        "Returns system information",
    )

    assert left == right


def test_repr_contains_name() -> None:
    schema = ToolSchema(
        "system_info",
        "Returns system information",
    )

    assert "system_info" in repr(schema)


def test_repr_contains_description() -> None:
    schema = ToolSchema(
        "system_info",
        "Returns system information",
    )

    assert "Returns system information" in repr(schema)


def test_supports_unicode() -> None:
    schema = ToolSchema(
        name="माहिती",
        description="सिस्टम माहिती",
    )

    assert schema.name == "माहिती"


def test_parameter_order_is_preserved() -> None:
    first = ToolParameter(
        "symbol",
        "Trading pair",
    )

    second = ToolParameter(
        "exchange",
        "Exchange",
    )

    schema = ToolSchema(
        name="price",
        description="Returns latest price",
        parameters=(first, second),
    )

    assert schema.parameters[0].name == "symbol"
    assert schema.parameters[1].name == "exchange"