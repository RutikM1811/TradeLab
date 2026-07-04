from app.types.model_result import ModelResult


def test_successful_model_result() -> None:
    result = ModelResult.ok(
        content="BTC is currently in an uptrend."
    )

    assert result.success is True
    assert result.content == "BTC is currently in an uptrend."
    assert result.error is None
    assert result.metadata == {}


def test_failed_model_result() -> None:
    result = ModelResult.fail(
        error="Model provider unavailable."
    )

    assert result.success is False
    assert result.content is None
    assert result.error == "Model provider unavailable."
    assert result.metadata == {}


def test_model_result_supports_metadata() -> None:
    result = ModelResult.ok(
        content="Analysis complete.",
        metadata={
            "provider": "test_provider",
            "model": "test_model",
            "input_tokens": 120,
            "output_tokens": 40,
        },
    )

    assert result.metadata["provider"] == "test_provider"
    assert result.metadata["model"] == "test_model"
    assert result.metadata["input_tokens"] == 120
    assert result.metadata["output_tokens"] == 40