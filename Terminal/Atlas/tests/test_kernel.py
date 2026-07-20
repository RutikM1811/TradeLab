import pytest

from app.kernel.bootstrap import Kernel
from app.models.atlas.atlas_model import AtlasModel
from app.models.atlas.development_backend import DevelopmentBackend
from app.models.atlas.fallback_backend import FallbackBackend
from app.models.atlas.groq_backend import GroqBackend
from app.models.atlas.openrouter_backend import OpenRouterBackend


@pytest.mark.anyio
async def test_kernel_executes_registered_system_info_tool() -> None:
    kernel = Kernel()
    kernel.boot()

    result = await kernel.execute_tool("system_info")

    assert result.success is True
    assert result.error is None
    assert result.data is not None
    assert "python_version" in result.data
    assert "operating_system" in result.data
    assert "platform" in result.data


def test_kernel_uses_development_backend() -> None:
    kernel = Kernel()
    kernel.settings.ATLAS_BACKEND = "development"

    kernel.boot()

    atlas_model = kernel.container.resolve(AtlasModel)

    assert isinstance(
        atlas_model._backend,
        DevelopmentBackend,
    )


def test_kernel_uses_openrouter_backend() -> None:
    kernel = Kernel()
    kernel.settings.ATLAS_BACKEND = "openrouter"
    kernel.settings.OPENROUTER_API_KEY = "test-key"

    kernel.boot()

    atlas_model = kernel.container.resolve(AtlasModel)

    assert isinstance(
        atlas_model._backend,
        OpenRouterBackend,
    )


def test_kernel_uses_groq_backend() -> None:
    kernel = Kernel()
    kernel.settings.ATLAS_BACKEND = "groq"
    kernel.settings.GROQ_API_KEY = "test-key"

    kernel.boot()

    atlas_model = kernel.container.resolve(AtlasModel)

    assert isinstance(
        atlas_model._backend,
        GroqBackend,
    )


def test_kernel_rejects_unsupported_backend() -> None:
    kernel = Kernel()
    kernel.settings.ATLAS_BACKEND = "invalid"

    with pytest.raises(
            ValueError,
            match="Unsupported Atlas backend: invalid",
    ):
        kernel.boot()


def test_kernel_boot_can_be_called_twice() -> None:
    kernel = Kernel()

    kernel.boot()
    kernel.boot()

    atlas_model = kernel.container.resolve(AtlasModel)

    assert atlas_model is not None


def test_kernel_resolves_atlas_model_after_boot() -> None:
    kernel = Kernel()

    kernel.boot()

    model = kernel.container.resolve(AtlasModel)

    assert isinstance(model, AtlasModel)


def test_kernel_returns_same_model_instance() -> None:
    kernel = Kernel()

    kernel.boot()

    first = kernel.container.resolve(AtlasModel)
    second = kernel.container.resolve(AtlasModel)

    assert first is second


def test_kernel_default_backend_is_valid() -> None:
    kernel = Kernel()

    kernel.boot()

    model = kernel.container.resolve(AtlasModel)

    assert model._backend is not None


def test_kernel_model_name_is_atlas() -> None:
    kernel = Kernel()

    kernel.boot()

    model = kernel.container.resolve(AtlasModel)

    assert model.name == "atlas"


def test_kernel_model_provider_is_atlas() -> None:
    kernel = Kernel()

    kernel.boot()

    model = kernel.container.resolve(AtlasModel)

    assert model.provider == "atlas"


def test_kernel_development_backend_name() -> None:
    kernel = Kernel()
    kernel.settings.ATLAS_BACKEND = "development"

    kernel.boot()

    model = kernel.container.resolve(AtlasModel)

    assert model._backend.name == "development"


def test_kernel_openrouter_backend_name() -> None:
    kernel = Kernel()
    kernel.settings.ATLAS_BACKEND = "openrouter"
    kernel.settings.OPENROUTER_API_KEY = "key"

    kernel.boot()

    model = kernel.container.resolve(AtlasModel)

    assert model._backend.name == "openrouter"


def test_kernel_groq_backend_name() -> None:
    kernel = Kernel()
    kernel.settings.ATLAS_BACKEND = "groq"
    kernel.settings.GROQ_API_KEY = "key"

    kernel.boot()

    model = kernel.container.resolve(AtlasModel)

    assert model._backend.name == "groq"


def test_kernel_fallback_backend_is_selected() -> None:
    kernel = Kernel()
    kernel.settings.ATLAS_BACKEND = "fallback"
    kernel.settings.GROQ_API_KEY = "g"
    kernel.settings.OPENROUTER_API_KEY = "o"

    kernel.boot()

    model = kernel.container.resolve(AtlasModel)

    assert isinstance(model._backend, FallbackBackend)


def test_kernel_fallback_backend_has_backends() -> None:
    kernel = Kernel()
    kernel.settings.ATLAS_BACKEND = "fallback"
    kernel.settings.GROQ_API_KEY = "g"
    kernel.settings.OPENROUTER_API_KEY = "o"

    kernel.boot()

    model = kernel.container.resolve(AtlasModel)

    assert len(model._backend.backends) > 0


@pytest.mark.anyio
async def test_system_info_tool_can_run_multiple_times() -> None:
    kernel = Kernel()

    kernel.boot()

    for _ in range(3):
        result = await kernel.execute_tool("system_info")
        assert result.success


@pytest.mark.anyio
async def test_system_info_tool_returns_dictionary() -> None:
    kernel = Kernel()

    kernel.boot()

    result = await kernel.execute_tool("system_info")

    assert isinstance(result.data, dict)


@pytest.mark.anyio
async def test_system_info_tool_python_version_is_string() -> None:
    kernel = Kernel()

    kernel.boot()

    result = await kernel.execute_tool("system_info")

    assert isinstance(result.data["python_version"], str)


@pytest.mark.anyio
async def test_system_info_tool_platform_is_string() -> None:
    kernel = Kernel()

    kernel.boot()

    result = await kernel.execute_tool("system_info")

    assert isinstance(result.data["platform"], str)


@pytest.mark.anyio
async def test_system_info_tool_operating_system_is_string() -> None:
    kernel = Kernel()

    kernel.boot()

    result = await kernel.execute_tool("system_info")

    assert isinstance(result.data["operating_system"], str)


def test_kernel_boot_does_not_change_backend_setting() -> None:
    kernel = Kernel()
    kernel.settings.ATLAS_BACKEND = "development"

    kernel.boot()

    assert kernel.settings.ATLAS_BACKEND == "development"


def test_kernel_container_exists_after_boot() -> None:
    kernel = Kernel()

    kernel.boot()

    assert kernel.container is not None


def test_kernel_settings_exist() -> None:
    kernel = Kernel()

    assert kernel.settings is not None


def test_kernel_can_boot_with_development_backend_multiple_times() -> None:
    kernel = Kernel()
    kernel.settings.ATLAS_BACKEND = "development"

    kernel.boot()
    kernel.boot()

    model = kernel.container.resolve(AtlasModel)

    assert isinstance(model._backend, DevelopmentBackend)


def test_kernel_boot_with_openrouter_multiple_times() -> None:
    kernel = Kernel()
    kernel.settings.ATLAS_BACKEND = "openrouter"
    kernel.settings.OPENROUTER_API_KEY = "key"

    kernel.boot()
    kernel.boot()

    model = kernel.container.resolve(AtlasModel)

    assert isinstance(model._backend, OpenRouterBackend)


def test_kernel_boot_with_groq_multiple_times() -> None:
    kernel = Kernel()
    kernel.settings.ATLAS_BACKEND = "groq"
    kernel.settings.GROQ_API_KEY = "key"

    kernel.boot()
    kernel.boot()

    model = kernel.container.resolve(AtlasModel)

    assert isinstance(model._backend, GroqBackend)


def test_kernel_default_model_is_atlas_model() -> None:
    kernel = Kernel()

    kernel.boot()

    assert isinstance(
        kernel.container.resolve(AtlasModel),
        AtlasModel,
    )


def test_kernel_boot_produces_backend_with_name() -> None:
    kernel = Kernel()

    kernel.boot()

    model = kernel.container.resolve(AtlasModel)

    assert isinstance(model._backend.name, str)


def test_kernel_backend_name_is_not_empty() -> None:
    kernel = Kernel()

    kernel.boot()

    model = kernel.container.resolve(AtlasModel)

    assert model._backend.name != ""