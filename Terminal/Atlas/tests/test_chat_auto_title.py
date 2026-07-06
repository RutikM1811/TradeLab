from typing import Any

import pytest

from app.memory.conversation import Conversation
from app.models.atlas.atlas_model import AtlasModel
from app.models.atlas.backend import AbstractInferenceBackend
from app.services.chat_runtime import ChatRuntime
from app.types.model_result import ModelResult


class AutoTitleBackend(AbstractInferenceBackend):
    @property
    def name(self) -> str:
        return "auto_title_backend"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        return ModelResult.ok(
            content="Atlas response."
        )


def create_runtime() -> ChatRuntime:
    return ChatRuntime(
        AtlasModel(AutoTitleBackend())
    )


@pytest.mark.anyio
async def test_first_user_message_creates_title() -> None:
    conversation = Conversation()
    runtime = create_runtime()

    await runtime.send(
        conversation,
        "Analyze BTC",
    )

    assert conversation.metadata.title == "Analyze BTC"


@pytest.mark.anyio
async def test_title_strips_surrounding_whitespace() -> None:
    conversation = Conversation()
    runtime = create_runtime()

    await runtime.send(
        conversation,
        "   Analyze BTC   ",
    )

    assert conversation.metadata.title == "Analyze BTC"


@pytest.mark.anyio
async def test_title_collapses_internal_whitespace() -> None:
    conversation = Conversation()
    runtime = create_runtime()

    await runtime.send(
        conversation,
        "Analyze    BTC     on   4h",
    )

    assert conversation.metadata.title == "Analyze BTC on 4h"


@pytest.mark.anyio
async def test_short_title_is_not_truncated() -> None:
    conversation = Conversation()
    runtime = create_runtime()

    await runtime.send(
        conversation,
        "ETH Analysis",
    )

    assert conversation.metadata.title == "ETH Analysis"


@pytest.mark.anyio
async def test_long_title_is_truncated() -> None:
    conversation = Conversation()
    runtime = create_runtime()

    message = (
        "Analyze Bitcoin market structure and identify "
        "important support and resistance levels"
    )

    await runtime.send(
        conversation,
        message,
    )

    assert len(conversation.metadata.title) == 53
    assert conversation.metadata.title.endswith("...")


@pytest.mark.anyio
async def test_truncated_title_uses_first_fifty_characters() -> None:
    conversation = Conversation()
    runtime = create_runtime()

    message = "A" * 60

    await runtime.send(
        conversation,
        message,
    )

    assert conversation.metadata.title == (
            ("A" * 50) + "..."
    )


@pytest.mark.anyio
async def test_second_user_message_does_not_change_title() -> None:
    conversation = Conversation()
    runtime = create_runtime()

    await runtime.send(
        conversation,
        "First question",
    )

    await runtime.send(
        conversation,
        "Second question",
    )

    assert conversation.metadata.title == "First question"


@pytest.mark.anyio
async def test_manual_title_is_preserved() -> None:
    conversation = Conversation()
    conversation.metadata.rename("My Custom Chat")

    runtime = create_runtime()

    await runtime.send(
        conversation,
        "Analyze BTC",
    )

    assert conversation.metadata.title == "My Custom Chat"


@pytest.mark.anyio
async def test_existing_default_title_with_history_is_not_replaced() -> None:
    conversation = Conversation()

    conversation.add_system("You are Atlas.")

    runtime = create_runtime()

    await runtime.send(
        conversation,
        "Analyze BTC",
    )

    assert (
            conversation.metadata.title
            == "New Conversation"
    )


@pytest.mark.anyio
async def test_assistant_response_does_not_become_title() -> None:
    conversation = Conversation()
    runtime = create_runtime()

    await runtime.send(
        conversation,
        "Hello Atlas",
    )

    assert conversation.metadata.title == "Hello Atlas"
    assert (
            conversation.metadata.title
            != "Atlas response."
    )


@pytest.mark.anyio
async def test_failed_generation_still_keeps_user_title() -> None:
    class FailedBackend(AbstractInferenceBackend):
        @property
        def name(self) -> str:
            return "failed_backend"

        async def generate(
                self,
                prompt: str,
                **kwargs: Any,
        ) -> ModelResult:
            return ModelResult.fail(
                error="Generation failed."
            )

    conversation = Conversation()

    runtime = ChatRuntime(
        AtlasModel(FailedBackend())
    )

    result = await runtime.send(
        conversation,
        "Important question",
    )

    assert result.success is False
    assert (
            conversation.metadata.title
            == "Important question"
    )


def test_create_title_handles_exactly_fifty_characters() -> None:
    message = "A" * 50

    title = ChatRuntime._create_title(message)

    assert title == message
    assert len(title) == 50