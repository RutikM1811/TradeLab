"""
Atlas application settings.

Centralized, type-safe configuration loaded from environment variables.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Atlas application configuration."""

    APP_NAME: str = "Atlas"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    HOST: str = "127.0.0.1"
    PORT: int = 8000
    CONVERSATION_STORAGE_PATH: str = "data/conversations"

    CONVERSATION_STORAGE_PATH: str = "data/conversations"
    ATLAS_BACKEND: str = "development"

    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "openai/gpt-5.2"
    OPENROUTER_BASE_URL: str = (
        "https://openrouter.ai/api/v1"
    )

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-20b"
    GROQ_BASE_URL: str = (
        "https://api.groq.com/openai/v1"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the shared cached Atlas settings instance."""

    return Settings()