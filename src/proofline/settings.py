# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Typed application settings loaded from the environment and local ``.env`` file."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApplicationSettings(BaseSettings):
    """Runtime configuration; environment variables take precedence over ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr | None = None


def load_settings() -> ApplicationSettings:
    """Load configuration from the active project's ignored ``.env`` by default."""

    return ApplicationSettings()
