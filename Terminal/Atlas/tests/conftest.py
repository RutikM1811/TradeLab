import pytest

from app.config.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    yield
    get_settings.cache_clear()