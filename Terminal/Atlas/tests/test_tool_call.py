from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest

from app.tools.tool_call import ToolCall


def test_tool_name_stored() -> None:
    call = ToolCall("system_info")

    assert call.name == "system_info"


def test_kwargs_stored() -> None:
    call = ToolCall(
        "price",
        {
            "symbol": "BTCUSDT",
        },
    )

    assert call.kwargs == {
        "symbol": "BTCUSDT",
    }


def test_equality() -> None:
    assert ToolCall(
        "price",
        {"symbol": "BTCUSDT"},
    ) == ToolCall(
        "price",
        {"symbol": "BTCUSDT"},
    )


def test_frozen() -> None:
    call = ToolCall("system_info")

    with pytest.raises(FrozenInstanceError):
        call.name = "cpu"


def test_hashable() -> None:
    call = ToolCall("system_info")

    assert hash(call)


def test_empty_kwargs() -> None:
    call = ToolCall("system_info")

    assert call.kwargs == {}


def test_unicode_args() -> None:
    call = ToolCall(
        "translate",
        {
            "text": "नमस्कार 世界 🚀",
        },
    )

    assert call.kwargs["text"] == "नमस्कार 世界 🚀"


def test_repr() -> None:
    call = ToolCall("system_info")

    assert "ToolCall" in repr(call)


def test_immutable_dict_reference() -> None:
    kwargs = {
        "symbol": "BTCUSDT",
    }

    call = ToolCall(
        "price",
        kwargs,
    )

    kwargs["symbol"] = "ETHUSDT"

    assert call.kwargs["symbol"] == "ETHUSDT"


def test_dataclass_fields() -> None:
    names = [field.name for field in fields(ToolCall)]

    assert names == [
        "name",
        "kwargs",
    ]