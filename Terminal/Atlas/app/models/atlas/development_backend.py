"""
Atlas development inference backend.

Provides deterministic responses while the real inference backend
is not yet configured.
"""

from typing import Any

from app.models.atlas.backend import AbstractInferenceBackend
from app.types.model_result import ModelResult
from collections.abc import AsyncIterator

class DevelopmentBackend(AbstractInferenceBackend):
    """Deterministic backend used during Atlas development."""

    @property
    def name(self) -> str:
        return "development"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        """Return a deterministic development response."""

        return ModelResult.ok(
            content="Atlas development response.",
            metadata={
                "mode": "development",
            },
        )
    async def generate_stream(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream a deterministic development response.
        """

        response = "Atlas development response."

        for word in response.split():
            yield word + " "