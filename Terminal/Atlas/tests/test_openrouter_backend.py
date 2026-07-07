from types import SimpleNamespace

import pytest

from app.memory.conversation import Conversation
from app.models.atlas.openrouter_backend import OpenRouterBackend


def create_response(
        content: str | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content
                )
            )
        ]
    )


def test_backend_name_is_openrouter() -> None:
    backend = OpenRouterBackend(
        api_key="test-key",
        model="openai/gpt-5.2",
    )

    assert backend.name == "openrouter"


def test_backend_rejects_empty_api_key() -> None:
    with pytest.raises(
            ValueError,
            match="OpenRouter API key cannot be empty",
    ):
        OpenRouterBackend(
            api_key="",
            model="openai/gpt-5.2",
        )


def test_backend_rejects_whitespace_api_key() -> None:
    with pytest.raises(ValueError):
        OpenRouterBackend(
            api_key="   ",
            model="openai/gpt-5.2",
        )


def test_backend_rejects_empty_model() -> None:
    with pytest.raises(
            ValueError,
            match="OpenRouter model cannot be empty",
    ):
        OpenRouterBackend(
            api_key="test-key",
            model="",
        )


def test_backend_rejects_whitespace_model() -> None:
    with pytest.raises(ValueError):
        OpenRouterBackend(
            api_key="test-key",
            model="   ",
        )


def test_backend_strips_model_whitespace() -> None:
    backend = OpenRouterBackend(
        api_key="test-key",
        model="   openai/gpt-5.2   ",
    )

    assert backend._model == "openai/gpt-5.2"


@pytest.mark.anyio
async def test_generate_returns_successful_model_result(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = OpenRouterBackend(
        api_key="test-key",
        model="openai/gpt-5.2",
    )

    async def fake_create(**kwargs):
        return create_response(
            "Hello from OpenRouter"
        )

    monkeypatch.setattr(
        backend._client.chat.completions,
        "create",
        fake_create,
    )

    result = await backend.generate("Hello")

    assert result.success is True
    assert result.content == "Hello from OpenRouter"


@pytest.mark.anyio
async def test_generate_forwards_prompt(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = OpenRouterBackend(
        api_key="test-key",
        model="openai/gpt-5.2",
    )

    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return create_response("Response")

    monkeypatch.setattr(
        backend._client.chat.completions,
        "create",
        fake_create,
    )

    await backend.generate("Analyze BTC")

    assert captured["messages"] == [
        {
            "role": "user",
            "content": "Analyze BTC",
        }
    ]


@pytest.mark.anyio
async def test_generate_forwards_model(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = OpenRouterBackend(
        api_key="test-key",
        model="openai/gpt-5.2",
    )

    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return create_response("Response")

    monkeypatch.setattr(
        backend._client.chat.completions,
        "create",
        fake_create,
    )

    await backend.generate("Hello")

    assert captured["model"] == "openai/gpt-5.2"


@pytest.mark.anyio
async def test_generate_forwards_extra_options(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = OpenRouterBackend(
        api_key="test-key",
        model="openai/gpt-5.2",
    )

    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return create_response("Response")

    monkeypatch.setattr(
        backend._client.chat.completions,
        "create",
        fake_create,
    )

    await backend.generate(
        "Hello",
        temperature=0.2,
        max_tokens=100,
    )

    assert captured["temperature"] == 0.2
    assert captured["max_tokens"] == 100


@pytest.mark.anyio
async def test_generate_handles_empty_content(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = OpenRouterBackend(
        api_key="test-key",
        model="openai/gpt-5.2",
    )

    async def fake_create(**kwargs):
        return create_response("")

    monkeypatch.setattr(
        backend._client.chat.completions,
        "create",
        fake_create,
    )

    result = await backend.generate("Hello")

    assert result.success is False
    assert result.error == (
        "OpenRouter returned an empty response."
    )


@pytest.mark.anyio
async def test_generate_handles_none_content(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = OpenRouterBackend(
        api_key="test-key",
        model="openai/gpt-5.2",
    )

    async def fake_create(**kwargs):
        return create_response(None)

    monkeypatch.setattr(
        backend._client.chat.completions,
        "create",
        fake_create,
    )

    result = await backend.generate("Hello")

    assert result.success is False


@pytest.mark.anyio
async def test_generate_handles_api_failure(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = OpenRouterBackend(
        api_key="test-key",
        model="openai/gpt-5.2",
    )

    async def fake_create(**kwargs):
        raise RuntimeError("API unavailable")

    monkeypatch.setattr(
        backend._client.chat.completions,
        "create",
        fake_create,
    )

    result = await backend.generate("Hello")

    assert result.success is False
    assert result.error == (
        "OpenRouter generation failed: API unavailable"
    )


@pytest.mark.anyio
async def test_structured_generation_forwards_all_messages(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = OpenRouterBackend(
        api_key="test-key",
        model="openai/gpt-5.2",
    )

    conversation = Conversation()
    conversation.add_system("You are Atlas.")
    conversation.add_user("Hello")
    conversation.add_assistant("Hi")
    conversation.add_user("Analyze BTC")

    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return create_response("BTC analysis")

    monkeypatch.setattr(
        backend._client.chat.completions,
        "create",
        fake_create,
    )

    await backend.generate_from_conversation(
        conversation
    )

    assert captured["messages"] == [
        {
            "role": "system",
            "content": "You are Atlas.",
        },
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Hi",
        },
        {
            "role": "user",
            "content": "Analyze BTC",
        },
    ]


@pytest.mark.anyio
async def test_structured_generation_forwards_model(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = OpenRouterBackend(
        api_key="test-key",
        model="openai/gpt-5.2",
    )

    conversation = Conversation()
    conversation.add_user("Hello")

    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return create_response("Response")

    monkeypatch.setattr(
        backend._client.chat.completions,
        "create",
        fake_create,
    )

    await backend.generate_from_conversation(
        conversation
    )

    assert captured["model"] == "openai/gpt-5.2"


@pytest.mark.anyio
async def test_structured_generation_returns_success(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = OpenRouterBackend(
        api_key="test-key",
        model="openai/gpt-5.2",
    )

    conversation = Conversation()
    conversation.add_user("Hello")

    async def fake_create(**kwargs):
        return create_response("Hello from Atlas")

    monkeypatch.setattr(
        backend._client.chat.completions,
        "create",
        fake_create,
    )

    result = await backend.generate_from_conversation(
        conversation
    )

    assert result.success is True
    assert result.content == "Hello from Atlas"


@pytest.mark.anyio
async def test_structured_generation_handles_api_failure(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = OpenRouterBackend(
        api_key="test-key",
        model="openai/gpt-5.2",
    )

    conversation = Conversation()
    conversation.add_user("Hello")

    async def fake_create(**kwargs):
        raise RuntimeError("API unavailable")

    monkeypatch.setattr(
        backend._client.chat.completions,
        "create",
        fake_create,
    )

    result = await backend.generate_from_conversation(
        conversation
    )

    assert result.success is False
    assert result.error == (
        "OpenRouter generation failed: API unavailable"
    )
@pytest.mark.anyio
async def test_generate_formats_authentication_error(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = OpenRouterBackend(
        api_key="test-key",
        model="openai/gpt-5.2",
    )

    error = RuntimeError("Invalid API key")
    error.status_code = 401

    async def fake_create(**kwargs):
        raise error

    monkeypatch.setattr(
        backend._client.chat.completions,
        "create",
        fake_create,
    )

    result = await backend.generate("Hello")

    assert result.success is False
    assert result.error == (
        "OpenRouter authentication failed. "
        "Check your API key."
    )


@pytest.mark.anyio
async def test_generate_formats_credit_error(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = OpenRouterBackend(
        api_key="test-key",
        model="openai/gpt-5.2",
    )

    error = RuntimeError("Payment required")
    error.status_code = 402

    async def fake_create(**kwargs):
        raise error

    monkeypatch.setattr(
        backend._client.chat.completions,
        "create",
        fake_create,
    )

    result = await backend.generate("Hello")

    assert result.success is False
    assert result.error == (
        "OpenRouter credits are required "
        "for the selected model."
    )


@pytest.mark.anyio
async def test_generate_formats_rate_limit_error(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = OpenRouterBackend(
        api_key="test-key",
        model="openrouter/free",
    )

    error = RuntimeError("Rate limit exceeded")
    error.status_code = 429

    async def fake_create(**kwargs):
        raise error

    monkeypatch.setattr(
        backend._client.chat.completions,
        "create",
        fake_create,
    )

    result = await backend.generate("Hello")

    assert result.success is False
    assert result.error == (
        "OpenRouter is temporarily rate-limited. "
        "Please retry shortly."
    )