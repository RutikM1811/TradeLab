import pytest

from app.kernel.bootstrap import Kernel


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