from dataclasses import FrozenInstanceError

import pytest

from app.tools.tool_parameter import ToolParameter


def test_parameter_stores_name() -> None:
    parameter = ToolParameter(
        name="symbol",
        description="Trading pair",
    )

    assert parameter.name == "symbol"


def test_parameter_stores_description() -> None:
    parameter = ToolParameter(
        name="symbol",
        description="Trading pair",
    )

    assert parameter.description == "Trading pair"


def test_default_type_is_string() -> None:
    parameter = ToolParameter(
        name="symbol",
        description="Trading pair",
    )

    assert parameter.type == "string"


def test_default_required_is_true() -> None:
    parameter = ToolParameter(
        name="symbol",
        description="Trading pair",
    )

    assert parameter.required is True


def test_custom_type() -> None:
    parameter = ToolParameter(
        name="limit",
        description="Maximum rows",
        type="integer",
    )

    assert parameter.type == "integer"


def test_optional_parameter() -> None:
    parameter = ToolParameter(
        name="exchange",
        description="Exchange name",
        required=False,
    )

    assert parameter.required is False


def test_parameters_are_equal() -> None:
    left = ToolParameter(
        "symbol",
        "Trading pair",
    )

    right = ToolParameter(
        "symbol",
        "Trading pair",
    )

    assert left == right


def test_parameters_are_hashable() -> None:
    parameter = ToolParameter(
        "symbol",
        "Trading pair",
    )

    assert hash(parameter)


def test_parameter_is_frozen() -> None:
    parameter = ToolParameter(
        "symbol",
        "Trading pair",
    )

    with pytest.raises(FrozenInstanceError):
        parameter.name = "btc"


def test_supports_unicode() -> None:
    parameter = ToolParameter(
        name="चलन",
        description="क्रिप्टो चलन",
    )

    assert parameter.name == "चलन"


def test_repr_contains_name() -> None:
    parameter = ToolParameter(
        "symbol",
        "Trading pair",
    )

    assert "symbol" in repr(parameter)


def test_repr_contains_description() -> None:
    parameter = ToolParameter(
        "symbol",
        "Trading pair",
    )

    assert "Trading pair" in repr(parameter)