import pytest

from app.models.atlas.backend import (
    AbstractInferenceBackend,
    ModelResult,
)
from app.models.atlas.fallback_backend import FallbackBackend


class SuccessBackend(AbstractInferenceBackend):

    @property
    def name(self) -> str:
        return "success"

    async def generate(self, prompt: str, **kwargs) -> ModelResult:
        return ModelResult.ok("success")

    async def generate_from_conversation(
            self,
            conversation,
            system_prompt=None,
            **kwargs,
    ) -> ModelResult:
        return await self.generate("")


class FailureBackend(AbstractInferenceBackend):

    @property
    def name(self) -> str:
        return "failure"

    async def generate(self, prompt: str, **kwargs) -> ModelResult:
        raise RuntimeError("backend failed")

    async def generate_from_conversation(
            self,
            conversation,
            system_prompt=None,
            **kwargs,
    ) -> ModelResult:
        return await self.generate("")


@pytest.mark.anyio
async def test_primary_backend_is_used_when_successful():
    backend = FallbackBackend(
        [
            SuccessBackend(),
            FailureBackend(),
        ]
    )

    result = await backend.generate("hello")

    assert result.success is True
    assert result.content == "success"


@pytest.mark.anyio
async def test_secondary_backend_is_used_when_primary_fails():
    backend = FallbackBackend(
        [
            FailureBackend(),
            SuccessBackend(),
        ]
    )

    result = await backend.generate("hello")

    assert result.success is True
    assert result.content == "success"


@pytest.mark.anyio
async def test_final_error_is_raised_when_all_backends_fail():
    backend = FallbackBackend(
        [
            FailureBackend(),
            FailureBackend(),
        ]
    )

    with pytest.raises(RuntimeError, match="backend failed"):
        await backend.generate("hello")


@pytest.mark.anyio
async def test_conversation_uses_primary_backend():
    backend = FallbackBackend(
        [
            SuccessBackend(),
            FailureBackend(),
        ]
    )

    result = await backend.generate_from_conversation([])

    assert result.success is True
    assert result.content == "success"


@pytest.mark.anyio
async def test_conversation_falls_back_to_secondary_backend():
    backend = FallbackBackend(
        [
            FailureBackend(),
            SuccessBackend(),
        ]
    )

    result = await backend.generate_from_conversation([])

    assert result.success is True
    assert result.content == "success"


@pytest.mark.anyio
async def test_conversation_raises_when_all_backends_fail():
    backend = FallbackBackend(
        [
            FailureBackend(),
            FailureBackend(),
        ]
    )

    with pytest.raises(RuntimeError, match="backend failed"):
        await backend.generate_from_conversation([])