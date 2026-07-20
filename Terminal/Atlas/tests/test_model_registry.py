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


class SecondDummyModel(AbstractModel):
    @property
    def name(self) -> str:
        return "second_model"

    @property
    def provider(self) -> str:
        return "second_provider"

    async def generate(
            self,
            prompt: str,
            **kwargs: object,
    ) -> ModelResult:
        return ModelResult.ok(content="second")


class UnicodeModel(AbstractModel):
    @property
    def name(self) -> str:
        return "नमस्कार"

    @property
    def provider(self) -> str:
        return "unicode"

    async def generate(
            self,
            prompt: str,
            **kwargs: object,
    ) -> ModelResult:
        return ModelResult.ok(content=prompt)


def test_new_registry_is_empty() -> None:
    registry = ModelRegistry()

    assert registry.all() == ()


def test_new_registry_contains_no_models() -> None:
    registry = ModelRegistry()

    assert registry.contains("dummy_model") is False


def test_get_returns_same_instance() -> None:
    registry = ModelRegistry()
    model = DummyModel()

    registry.register(model)

    assert registry.get("dummy_model") is model


def test_contains_returns_true_after_registration() -> None:
    registry = ModelRegistry()

    registry.register(DummyModel())

    assert registry.contains("dummy_model")


def test_contains_returns_false_for_unknown_model() -> None:
    registry = ModelRegistry()

    assert not registry.contains("unknown")


def test_register_multiple_models() -> None:
    registry = ModelRegistry()

    registry.register(DummyModel())
    registry.register(SecondDummyModel())

    assert len(registry.all()) == 2


def test_all_returns_models_in_registration_order() -> None:
    registry = ModelRegistry()

    first = DummyModel()
    second = SecondDummyModel()

    registry.register(first)
    registry.register(second)

    assert registry.all() == (first, second)


def test_get_second_registered_model() -> None:
    registry = ModelRegistry()

    second = SecondDummyModel()

    registry.register(DummyModel())
    registry.register(second)

    assert registry.get("second_model") is second


def test_contains_second_registered_model() -> None:
    registry = ModelRegistry()

    registry.register(DummyModel())
    registry.register(SecondDummyModel())

    assert registry.contains("second_model")


def test_all_returns_tuple_when_empty() -> None:
    registry = ModelRegistry()

    assert isinstance(registry.all(), tuple)


def test_all_returns_tuple_when_populated() -> None:
    registry = ModelRegistry()

    registry.register(DummyModel())

    assert isinstance(registry.all(), tuple)


def test_register_unicode_model_name() -> None:
    registry = ModelRegistry()

    model = UnicodeModel()

    registry.register(model)

    assert registry.get("नमस्कार") is model


def test_contains_unicode_model_name() -> None:
    registry = ModelRegistry()

    registry.register(UnicodeModel())

    assert registry.contains("नमस्कार")


def test_registry_size_after_multiple_registrations() -> None:
    registry = ModelRegistry()

    registry.register(DummyModel())
    registry.register(SecondDummyModel())

    assert len(registry.all()) == 2


def test_get_preserves_object_identity() -> None:
    registry = ModelRegistry()

    model = DummyModel()

    registry.register(model)

    retrieved = registry.get("dummy_model")

    assert id(retrieved) == id(model)


def test_register_does_not_modify_model_name() -> None:
    registry = ModelRegistry()

    model = DummyModel()

    registry.register(model)

    assert model.name == "dummy_model"


def test_registry_can_store_different_providers() -> None:
    registry = ModelRegistry()

    first = DummyModel()
    second = SecondDummyModel()

    registry.register(first)
    registry.register(second)

    assert registry.get("dummy_model").provider == "dummy_provider"
    assert registry.get("second_model").provider == "second_provider"


def test_get_unknown_model_message() -> None:
    registry = ModelRegistry()

    with pytest.raises(
            KeyError,
            match="not registered",
    ):
        registry.get("abc")


def test_duplicate_registration_message() -> None:
    registry = ModelRegistry()

    registry.register(DummyModel())

    with pytest.raises(
            ValueError,
            match="already registered",
    ):
        registry.register(DummyModel())


def test_empty_name_message() -> None:
    registry = ModelRegistry()

    with pytest.raises(
            ValueError,
            match="Model name cannot be empty",
    ):
        registry.register(EmptyNameModel())


def test_registry_stores_exact_model_instance() -> None:
    registry = ModelRegistry()

    model = DummyModel()

    registry.register(model)

    assert registry.all()[0] is model


def test_registry_all_length_matches_registration_count() -> None:
    registry = ModelRegistry()

    registry.register(DummyModel())
    registry.register(SecondDummyModel())

    assert len(registry.all()) == 2