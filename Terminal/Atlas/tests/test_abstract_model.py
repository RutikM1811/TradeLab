from typing import Any

import pytest

from app.contracts.abstract_model import AbstractModel
from app.types.model_result import ModelResult


class ValidModel(AbstractModel):
    @property
    def name(self) -> str:
        return "test_model"

    @property
    def provider(self) -> str:
        return "test_provider"

    async def generate(
            self,
            prompt: str,
            **kwargs: Any,
    ) -> ModelResult:
        return ModelResult.ok(
            content=f"Response to: {prompt}",
            metadata={
                "provider": self.provider,
                "model": self.name,
            },
        )


class IncompleteModel(AbstractModel):
    pass


def test_incomplete_model_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        IncompleteModel()


def test_valid_model_properties() -> None:
    model = ValidModel()

    assert model.name == "test_model"
    assert model.provider == "test_provider"


@pytest.mark.anyio
async def test_valid_model_generates_result() -> None:
    model = ValidModel()

    result = await model.generate(
        "Analyze BTC"
    )

    assert result.success is True
    assert result.content == "Response to: Analyze BTC"
    assert result.error is None
    assert result.metadata["provider"] == "test_provider"
    assert result.metadata["model"] == "test_model"