import pytest

from app.kernel.bootstrap import Kernel
from app.models.atlas.atlas_model import AtlasModel
from app.models.atlas.development_backend import DevelopmentBackend
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