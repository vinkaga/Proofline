# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Vinay Agarwal
"""Verify local developer settings are typed and never represented as plain values."""

from proofline.settings import load_settings


def test_settings_load_openai_key_from_dotenv(tmp_path, monkeypatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENAI_API_KEY=dotenv-key\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    settings = load_settings()

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "dotenv-key"
    assert "dotenv-key" not in repr(settings)


def test_environment_overrides_dotenv(tmp_path, monkeypatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENAI_API_KEY=dotenv-key\n")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    monkeypatch.chdir(tmp_path)

    settings = load_settings()

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "environment-key"
