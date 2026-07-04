from app.types.tool_result import ToolResult


def test_successful_tool_result() -> None:
    result = ToolResult.ok(
        data={"symbol": "BTCUSDT", "price": 65000}
    )

    assert result.success is True
    assert result.data == {
        "symbol": "BTCUSDT",
        "price": 65000,
    }
    assert result.error is None
    assert result.metadata == {}


def test_failed_tool_result() -> None:
    result = ToolResult.fail(
        error="Market data provider unavailable"
    )

    assert result.success is False
    assert result.data is None
    assert result.error == "Market data provider unavailable"
    assert result.metadata == {}


def test_tool_result_supports_metadata() -> None:
    result = ToolResult.ok(
        data={"price": 65000},
        metadata={
            "provider": "binance",
            "symbol": "BTCUSDT",
        },
    )

    assert result.metadata["provider"] == "binance"
    assert result.metadata["symbol"] == "BTCUSDT"