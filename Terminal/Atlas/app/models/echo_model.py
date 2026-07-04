"""
Atlas Echo Model.

A built-in model used to verify the complete model runtime pipeline
without depending on an external AI provider.
"""

from typing import Any

from app.contracts.abstract_model import AbstractModel
from app.types.model_result import ModelResult


class EchoModel(AbstractModel):
    """Return the supplied prompt as the generated response."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def provider(self) -> str:
        return "atlas"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        """Return the prompt as a successful model result."""

        return ModelResult.ok(
            content=prompt,
            metadata={
                "provider": self.provider,
                "model": self.name,
            },
        )