from app.tools.tool_call import ToolCall
from app.tools.tool_call_parser import ToolCallParser


def test_parse_tool_without_kwargs():
    parser = ToolCallParser()

    result = parser.parse(
        """
CALL_TOOL:
system_info
"""
    )

    assert result == ToolCall(
        "system_info",
        {},
    )


def test_parse_tool_with_kwargs():
    parser = ToolCallParser()

    result = parser.parse(
        """
CALL_TOOL:
price
symbol=BTCUSDT
"""
    )

    assert result == ToolCall(
        "price",
        {
            "symbol": "BTCUSDT",
        },
    )


def test_returns_none_for_normal_text():
    parser = ToolCallParser()

    assert parser.parse("Hello World") is None


def test_returns_none_when_missing_tool():
    parser = ToolCallParser()

    assert parser.parse("CALL_TOOL:") is None


def test_multiple_kwargs():
    parser = ToolCallParser()

    result = parser.parse(
        """
CALL_TOOL:
price
symbol=BTCUSDT
exchange=BINANCE
"""
    )

    assert result == ToolCall(
        "price",
        {
            "symbol": "BTCUSDT",
            "exchange": "BINANCE",
        },
    )