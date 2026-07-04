import pytest

from app.contracts.abstract_model import AbstractModel
from app.models.model_registry import ModelRegistry
from app.types.model_result import ModelResult


class DummyModel(AbstractModel):
    @property
    def name(self) -> str:
        return "dummy_model"

    @property
    def provider(self) -> str:
        return "dummy_provider"

    async def generate(
            self,
            prompt: str,
            **kwargs: object,
    ) -> ModelResult:
        return ModelResult.ok(content=f"Response to: {prompt}")


class EmptyNameModel(DummyModel):
    @property
    def name(self) -> str:
        return "   "


def test_model_registry_registers_and_gets_model() -> None:
    registry = ModelRegistry()
    model = DummyModel()

    registry.register(model)

    assert registry.get("dummy_model") is model
    assert registry.contains("dummy_model")


def test_model_registry_rejects_duplicate_model() -> None:
    registry = ModelRegistry()

    registry.register(DummyModel())

    with pytest.raises(ValueError):
        registry.register(DummyModel())


def test_model_registry_rejects_empty_model_name() -> None:
    registry = ModelRegistry()

    with pytest.raises(ValueError):
        registry.register(EmptyNameModel())


def test_model_registry_raises_for_missing_model() -> None:
    registry = ModelRegistry()

    with pytest.raises(KeyError):
        registry.get("missing_model")


def test_model_registry_returns_all_models() -> None:
    registry = ModelRegistry()
    model = DummyModel()

    registry.register(model)

    models = registry.all()

    assert models == (model,)
    assert isinstance(models, tuple)