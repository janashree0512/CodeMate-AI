"""
Unit tests for .env key loading, sanitization, and specific error validation.
Uses temporary .env files and monkeypatching without real API credentials.
"""

from pathlib import Path
import pytest

from config import (
    Config,
    load_config,
    ConfigurationError,
    validate_api_key_string,
    mask_key,
    PLACEHOLDER_KEY,
    DEFAULT_MODEL,
)


class TestEnvLoadingAndKeyValidation:
    """Tests for absolute path loading, whitespace/BOM/quote stripping, and error diagnostics."""

    def test_mask_key_formatting(self):
        assert mask_key("") == "[NOT SET]"
        assert mask_key(PLACEHOLDER_KEY) == "[NOT SET]"
        assert mask_key("short") == "[SET]"
        assert mask_key("gsk_1234567890abcdef") == "gsk_...cdef"

    def test_missing_env_file_raises_specific_error(self, tmp_path):
        nonexistent = tmp_path / "missing.env"
        with pytest.raises(ConfigurationError) as exc:
            validate_api_key_string("gsk_validkey1234", env_path=nonexistent)
        assert ".env not found at" in str(exc.value)

    def test_placeholder_or_empty_key_raises_specific_error(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("GROQ_API_KEY=\n", encoding="utf-8")

        with pytest.raises(ConfigurationError) as exc:
            validate_api_key_string("", env_path=env_file)
        assert "GROQ_API_KEY is empty or still the placeholder" in str(exc.value)

        with pytest.raises(ConfigurationError) as exc2:
            validate_api_key_string(PLACEHOLDER_KEY, env_path=env_file)
        assert "GROQ_API_KEY is empty or still the placeholder" in str(exc2.value)

    def test_invalid_prefix_raises_specific_error(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("GROQ_API_KEY=sk_wrongprefix1234\n", encoding="utf-8")

        with pytest.raises(ConfigurationError) as exc:
            validate_api_key_string("sk_wrongprefix1234", env_path=env_file)
        assert "Key should start with gsk_" in str(exc.value)
        assert "check for a copy/paste mistake" in str(exc.value)

    def test_sanitization_of_quotes_spaces_carriage_return_bom(self, tmp_path, monkeypatch):
        # Create a .env file with BOM, quotes, trailing spaces, and \r
        env_file = tmp_path / ".env"
        dirty_content = f'\ufeffGROQ_API_KEY="  gsk_sampletestkey9876  "\r\nMODEL_NAME=\'  {DEFAULT_MODEL}  \'\r\n'
        env_file.write_text(dirty_content, encoding="utf-8")

        cfg = load_config(env_file=env_file, strict_api_key=False)
        assert cfg.groq_api_key == "gsk_sampletestkey9876"
        assert cfg.model_name == DEFAULT_MODEL
        assert not cfg.groq_api_key.endswith("\r")
        assert not cfg.groq_api_key.startswith('"')

    def test_config_strict_validation_passes_valid_key(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("GROQ_API_KEY=gsk_validtestkeyabcdef1234\n", encoding="utf-8")

        cfg = load_config(env_file=env_file, strict_api_key=True)
        assert cfg.groq_api_key.startswith("gsk_")
        assert cfg.groq_api_key == "gsk_validtestkeyabcdef1234"

    def test_load_config_missing_env_file_raises_specific_error(self, tmp_path):
        missing_file = tmp_path / "does_not_exist.env"
        with pytest.raises(ConfigurationError) as exc:
            load_config(env_file=missing_file, strict_api_key=False)
        assert ".env not found at" in str(exc.value)
